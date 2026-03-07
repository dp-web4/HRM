#!/usr/bin/env python3
"""
ACT Chain Client - HTTP client for ACT blockchain REST API.

Provides SAGE with the ability to register LCTs, sync trust tensors,
and query chain state via the Cosmos SDK REST gateway.

All methods are non-blocking with timeout and graceful fallback:
- Returns None on any failure (network, timeout, parse error)
- Logs warnings but never raises exceptions to caller
- SAGE must function normally when chain is unreachable

Endpoints (Cosmos SDK REST gateway on :1317):
- GET  /racecar-web/lctmanager/v1/get_lct/{lct_id}
- GET  /racecar-web/lctmanager/v1/get_component_relationships/{component_id}
- POST /cosmos/tx/v1beta1/txs  (broadcast transactions)
- GET  /cosmos/base/tendermint/v1beta1/node_info  (health check)
"""

import json
import urllib.request
import urllib.error
import urllib.parse
from typing import Optional, Dict, Any


# Default timeout for all chain requests (seconds)
_DEFAULT_TIMEOUT = 5


class ACTChainClient:
    """
    HTTP client for ACT blockchain REST API.

    All public methods return Optional types and never raise.
    Chain unavailability is a normal operating condition.
    """

    def __init__(self, base_url: str = "http://localhost:1317", timeout: int = _DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str) -> Optional[Dict[str, Any]]:
        """GET request to chain. Returns parsed JSON or None."""
        url = f"{self.base_url}{path}"
        try:
            req = urllib.request.Request(url, method="GET")
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                ConnectionError, json.JSONDecodeError, OSError) as e:
            print(f"  [chain] GET {path} failed: {e}")
            return None

    def _post(self, path: str, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """POST request to chain. Returns parsed JSON or None."""
        url = f"{self.base_url}{path}"
        try:
            data = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(url, data=data, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                ConnectionError, json.JSONDecodeError, OSError) as e:
            print(f"  [chain] POST {path} failed: {e}")
            return None

    # ── Health ──────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Check if ACT chain is reachable. Returns True/False, never raises."""
        result = self._get("/cosmos/base/tendermint/v1beta1/node_info")
        return result is not None

    def get_node_info(self) -> Optional[Dict[str, Any]]:
        """Get chain node info (network, version, etc.)."""
        return self._get("/cosmos/base/tendermint/v1beta1/node_info")

    # ── LCT Operations ─────────────────────────────────────────

    def get_lct(self, lct_id: str) -> Optional[Dict[str, Any]]:
        """Query an LCT by ID. Returns LCT data or None if not found/unreachable."""
        encoded = urllib.parse.quote(lct_id, safe="")
        return self._get(f"/racecar-web/lctmanager/v1/get_lct/{encoded}")

    def get_component_relationships(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Query all LCT relationships for a component."""
        encoded = urllib.parse.quote(component_id, safe="")
        return self._get(f"/racecar-web/lctmanager/v1/get_component_relationships/{encoded}")

    def mint_lct(
        self,
        entity_name: str,
        entity_type: str,
        t3_tensor: Dict[str, float],
        v3_tensor: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, str]] = None,
        initial_adp: str = "1000",
        sender: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Mint a new LCT on the ACT chain.

        This broadcasts a MintLCT transaction via the Cosmos SDK tx broadcast endpoint.
        The sender must have a funded account in the chain's keyring.

        Args:
            entity_name: Human-readable name (e.g. "SAGE-legion")
            entity_type: Entity type (e.g. "agent", "human", "device")
            t3_tensor: Trust tensor {talent, training, temperament}
            v3_tensor: Value tensor {valuation, veracity, validity} (optional)
            metadata: Key-value metadata pairs
            initial_adp: Initial ADP allocation (as string decimal)
            sender: Cosmos account address of the sender

        Returns:
            Transaction response dict, or None on failure
        """
        # Format T3 as the CLI expects: [talent, training, temperament]
        t3_list = [
            str(t3_tensor.get("talent", 0.5)),
            str(t3_tensor.get("training", 0.5)),
            str(t3_tensor.get("temperament", 0.5)),
        ]

        v3_list = None
        if v3_tensor:
            v3_list = [
                str(v3_tensor.get("valuation", 0.5)),
                str(v3_tensor.get("veracity", 0.5)),
                str(v3_tensor.get("validity", 0.5)),
            ]

        # Build the message body for Cosmos SDK tx broadcast
        msg = {
            "@type": "/racecarweb.lctmanager.v1.MsgMintLCT",
            "creator": sender,
            "entity_name": entity_name,
            "entity_type": entity_type,
            "initial_adp_amount": initial_adp,
            "initial_t3_tensor": t3_list,
        }
        if v3_list:
            msg["initial_v3_tensor"] = v3_list
        if metadata:
            msg["metadata"] = metadata

        # Broadcast via Cosmos tx endpoint
        tx_body = {
            "tx": {
                "body": {
                    "messages": [msg],
                },
                "auth_info": {
                    "fee": {"amount": [{"denom": "stake", "amount": "100"}], "gas_limit": "200000"},
                },
            },
            "mode": "BROADCAST_MODE_SYNC",
        }

        return self._post("/cosmos/tx/v1beta1/txs", tx_body)

    # ── Trust Tensor Operations ────────────────────────────────

    def update_trust_tensor(
        self,
        lct_id: str,
        t3_tensor: Dict[str, float],
        context: str = "",
        evidence_hash: str = "",
        sender: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Update trust tensor for an LCT on chain.

        Note: This requires MsgUpdateTrustTensor to be implemented in ACT's
        trusttensor module. Until then, this will fail gracefully (returns None).

        Args:
            lct_id: The LCT whose trust to update
            t3_tensor: New T3 values {talent, training, temperament}
            context: What triggered the update (e.g. "session_end")
            evidence_hash: Hash of supporting evidence
            sender: Cosmos account address

        Returns:
            Transaction response or None
        """
        msg = {
            "@type": "/racecarweb.trusttensor.v1.MsgUpdateTrustTensor",
            "creator": sender,
            "lct_id": lct_id,
            "talent": str(t3_tensor.get("talent", 0.5)),
            "training": str(t3_tensor.get("training", 0.5)),
            "temperament": str(t3_tensor.get("temperament", 0.5)),
            "context": context,
            "evidence_hash": evidence_hash,
        }

        tx_body = {
            "tx": {
                "body": {
                    "messages": [msg],
                },
                "auth_info": {
                    "fee": {"amount": [{"denom": "stake", "amount": "100"}], "gas_limit": "200000"},
                },
            },
            "mode": "BROADCAST_MODE_SYNC",
        }

        return self._post("/cosmos/tx/v1beta1/txs", tx_body)

    def get_trust_history(self, lct_id: str, limit: int = 10) -> Optional[Dict[str, Any]]:
        """
        Query trust tensor history for an LCT.

        Note: Requires GetTrustHistory query in ACT's trusttensor module.
        Until implemented, returns None gracefully.
        """
        encoded = urllib.parse.quote(lct_id, safe="")
        return self._get(f"/racecar-web/trusttensor/v1/trust_history/{encoded}?limit={limit}")

    # ── Account Operations ─────────────────────────────────────

    def get_account(self, address: str) -> Optional[Dict[str, Any]]:
        """Query account info (balance, sequence number)."""
        return self._get(f"/cosmos/auth/v1beta1/accounts/{address}")

    def get_balance(self, address: str) -> Optional[Dict[str, Any]]:
        """Query account balance."""
        return self._get(f"/cosmos/bank/v1beta1/balances/{address}")
