#!/usr/bin/env python3
"""
SAGE ATP/ADP Integration Wrapper
Society 4 Economic Integration for Genesis SAGE v0.1

Wraps SAGE model with ATP/ADP energy economy tracking.
Implements LAW-ECON-001, LAW-ECON-003, PROC-ATP-DISCHARGE.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

@dataclass
class AtpConfig:
    """ATP/ADP configuration for SAGE"""
    # Initial allocation (LAW-ECON-001)
    initial_atp: int = 200
    daily_recharge: int = 20

    # Cost structure (PROC-ATP-DISCHARGE)
    l_level_cost: int = 1      # Tactical attention (cheap)
    h_level_cost: int = 5      # Strategic attention (expensive)
    consciousness_cost: int = 2 # KV-cache access
    training_cost: int = 10    # Training step
    validation_cost: int = 3   # Validation run

    # Refund structure (efficiency incentives)
    excellent_refund: float = 0.5  # 50% refund for excellent reasoning
    good_refund: float = 0.25      # 25% refund for good reasoning
    efficient_refund: float = 0.3  # 30% bonus for efficiency

    # Economic limits
    min_atp_for_h_level: int = 10  # Minimum ATP to use strategic
    emergency_reserve: int = 5     # Always keep this much

    # Society identity
    role_lct: str = "lct:web4:society:federation:sage_model"
    society_lct: str = "lct:web4:society:federation"


class AtpTransaction:
    """ATP transaction record"""

    def __init__(self, tx_type: str, amount: int, balance_before: int,
                 balance_after: int, reason: str, metadata: Dict = None):
        self.tx_type = tx_type  # "discharge", "recharge", "refund"
        self.amount = amount
        self.balance_before = balance_before
        self.balance_after = balance_after
        self.reason = reason
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        return {
            "type": self.tx_type,
            "amount": self.amount,
            "balance_before": self.balance_before,
            "balance_after": self.balance_after,
            "reason": self.reason,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class AtpPool:
    """Simplified ATP/ADP pool for SAGE model"""

    def __init__(self, config: AtpConfig, state_file: Optional[Path] = None):
        self.config = config
        self.state_file = state_file or Path("sage_atp_state.json")

        # State
        self.atp_balance: int = config.initial_atp
        self.adp_balance: int = 0
        self.last_recharge: datetime = datetime.now(timezone.utc)
        self.transactions: list[AtpTransaction] = []

        # Statistics
        self.total_discharged: int = 0
        self.total_recharged: int = 0
        self.total_refunded: int = 0

        # Load existing state if available
        self.load_state()

    def discharge(self, amount: int, reason: str, metadata: Dict = None) -> bool:
        """
        Discharge ATP (consume energy)
        Returns True if successful, False if insufficient ATP
        """
        if amount > self.atp_balance:
            return False

        balance_before = self.atp_balance
        self.atp_balance -= amount
        self.adp_balance += amount
        self.total_discharged += amount

        tx = AtpTransaction(
            tx_type="discharge",
            amount=amount,
            balance_before=balance_before,
            balance_after=self.atp_balance,
            reason=reason,
            metadata=metadata
        )
        self.transactions.append(tx)

        return True

    def refund(self, amount: int, reason: str, metadata: Dict = None) -> None:
        """
        Refund ATP (reward for good reasoning)
        Converts ADP back to ATP
        """
        actual_refund = min(amount, self.adp_balance)

        balance_before = self.atp_balance
        self.atp_balance += actual_refund
        self.adp_balance -= actual_refund
        self.total_refunded += actual_refund

        tx = AtpTransaction(
            tx_type="refund",
            amount=actual_refund,
            balance_before=balance_before,
            balance_after=self.atp_balance,
            reason=reason,
            metadata=metadata
        )
        self.transactions.append(tx)

    def daily_recharge(self) -> int:
        """
        Daily ATP recharge (LAW-ECON-003)
        Returns amount recharged
        """
        now = datetime.now(timezone.utc)
        day_boundary = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Check if day boundary crossed
        if self.last_recharge < day_boundary <= now:
            # Calculate recharge amount (capped at initial allocation)
            recharge_amount = min(
                self.config.daily_recharge,
                self.config.initial_atp - self.atp_balance
            )

            if recharge_amount > 0:
                balance_before = self.atp_balance
                self.atp_balance += recharge_amount
                self.total_recharged += recharge_amount
                self.last_recharge = now

                tx = AtpTransaction(
                    tx_type="recharge",
                    amount=recharge_amount,
                    balance_before=balance_before,
                    balance_after=self.atp_balance,
                    reason="Daily 00:00 UTC regeneration per LAW-ECON-003",
                    metadata={"day": day_boundary.isoformat()}
                )
                self.transactions.append(tx)

                return recharge_amount

        return 0

    def get_balance(self) -> Tuple[int, int]:
        """Get current ATP and ADP balances"""
        return self.atp_balance, self.adp_balance

    def get_statistics(self) -> Dict:
        """Get pool statistics"""
        return {
            "atp_balance": self.atp_balance,
            "adp_balance": self.adp_balance,
            "total_discharged": self.total_discharged,
            "total_recharged": self.total_recharged,
            "total_refunded": self.total_refunded,
            "efficiency": self.total_refunded / max(self.total_discharged, 1),
            "transaction_count": len(self.transactions),
            "last_recharge": self.last_recharge.isoformat()
        }

    def save_state(self) -> None:
        """Save ATP pool state to file"""
        state = {
            "atp_balance": self.atp_balance,
            "adp_balance": self.adp_balance,
            "last_recharge": self.last_recharge.isoformat(),
            "total_discharged": self.total_discharged,
            "total_recharged": self.total_recharged,
            "total_refunded": self.total_refunded,
            "transactions": [tx.to_dict() for tx in self.transactions[-100:]]  # Keep last 100
        }

        self.state_file.write_text(json.dumps(state, indent=2))

    def load_state(self) -> None:
        """Load ATP pool state from file"""
        if not self.state_file.exists():
            return

        state = json.loads(self.state_file.read_text())
        self.atp_balance = state["atp_balance"]
        self.adp_balance = state["adp_balance"]
        self.last_recharge = datetime.fromisoformat(state["last_recharge"])
        self.total_discharged = state.get("total_discharged", 0)
        self.total_recharged = state.get("total_recharged", 0)
        self.total_refunded = state.get("total_refunded", 0)


class SAGEWithEconomy(nn.Module):
    """
    SAGE model wrapped with ATP/ADP energy economy
    Society 4's economic integration for Genesis SAGE v0.1
    """

    def __init__(self, sage_model, atp_config: Optional[AtpConfig] = None,
                 state_file: Optional[Path] = None):
        super().__init__()

        self.sage = sage_model
        self.config = atp_config or AtpConfig()
        self.atp_pool = AtpPool(self.config, state_file)

        # Training mode tracking
        self.training_mode = False
        self.inference_count = 0
        self.training_step_count = 0

    def forward(self, input_ids: torch.Tensor,
                use_consciousness: bool = True,
                record_economics: bool = True) -> Dict:
        """
        Forward pass with ATP/ADP tracking

        Args:
            input_ids: Input token IDs
            use_consciousness: Whether to use consciousness cache
            record_economics: Whether to record ATP costs

        Returns:
            Dict with model output + economic metadata
        """
        # Check daily recharge
        self.atp_pool.daily_recharge()

        # Get current ATP balance
        atp_balance, adp_balance = self.atp_pool.get_balance()

        # Determine if we can afford H-level
        can_use_h_level = atp_balance >= self.config.min_atp_for_h_level

        if not can_use_h_level:
            # Low energy: Force L-level only
            use_consciousness = False
            # Temporarily disable H-level in SAGE (if configurable)
            # For now, just flag it

        # Run SAGE forward pass
        sage_output = self.sage(input_ids, use_consciousness=use_consciousness)

        # Calculate ATP cost
        atp_cost = self._calculate_atp_cost(sage_output)

        # Record discharge if requested
        economics_metadata = {}
        if record_economics:
            success = self.atp_pool.discharge(
                amount=atp_cost,
                reason=f"SAGE inference #{self.inference_count}",
                metadata={
                    "h_ratio": sage_output.get('h_ratio', 0),
                    "consciousness_size": sage_output.get('consciousness_size', 0),
                    "input_length": input_ids.size(1)
                }
            )

            if not success:
                economics_metadata["warning"] = "Insufficient ATP - operation allowed but not charged"

            self.inference_count += 1
            economics_metadata.update({
                "atp_cost": atp_cost,
                "atp_balance": self.atp_pool.atp_balance,
                "adp_balance": self.atp_pool.adp_balance,
                "can_use_h_level": can_use_h_level
            })

        # Combine SAGE output with economic metadata
        output = {
            **sage_output,
            "economics": economics_metadata
        }

        return output

    def _calculate_atp_cost(self, sage_output: Dict) -> int:
        """Calculate ATP cost based on computation"""
        if self.training_mode:
            return self.config.training_cost

        # Base L-level cost
        cost = self.config.l_level_cost

        # H-level usage (proportional to ratio)
        h_ratio = sage_output.get('h_ratio', 0)
        if h_ratio > 0:
            h_cost = int(h_ratio * self.config.h_level_cost)
            cost += h_cost

        # Consciousness cache access
        consciousness_size = sage_output.get('consciousness_size', 0)
        if consciousness_size > 0:
            cache_cost = min(consciousness_size // 100, self.config.consciousness_cost)
            cost += cache_cost

        return max(cost, 1)  # Minimum 1 ATP

    def record_training_step(self, loss: float, metrics: Dict) -> None:
        """Record ATP cost for training step"""
        self.training_step_count += 1

        self.atp_pool.discharge(
            amount=self.config.training_cost,
            reason=f"Training step #{self.training_step_count}",
            metadata={
                "loss": float(loss),
                **metrics
            }
        )

    def reward_reasoning(self, reward_score: float, atp_spent: int,
                        efficiency: float = 0.0) -> int:
        """
        Reward good reasoning with ATP refund

        Args:
            reward_score: Reasoning quality (0-1)
            atp_spent: ATP spent on this inference
            efficiency: Efficiency bonus (0-1)

        Returns:
            Amount of ATP refunded
        """
        refund_amount = 0

        # Base refund on reasoning quality
        if reward_score > 0.8:
            refund_amount = int(atp_spent * self.config.excellent_refund)
        elif reward_score > 0.5:
            refund_amount = int(atp_spent * self.config.good_refund)

        # Efficiency bonus
        if efficiency > 0.7:
            efficiency_bonus = int(refund_amount * self.config.efficient_refund)
            refund_amount += efficiency_bonus

        # Apply refund
        if refund_amount > 0:
            self.atp_pool.refund(
                amount=refund_amount,
                reason="Reasoning quality reward",
                metadata={
                    "reward_score": reward_score,
                    "efficiency": efficiency,
                    "atp_spent": atp_spent
                }
            )

        return refund_amount

    def get_economics_report(self) -> Dict:
        """Get comprehensive economics report"""
        stats = self.atp_pool.get_statistics()

        return {
            "pool_statistics": stats,
            "inference_count": self.inference_count,
            "training_steps": self.training_step_count,
            "average_cost_per_inference": stats["total_discharged"] / max(self.inference_count, 1),
            "role_lct": self.config.role_lct,
            "society_lct": self.config.society_lct
        }

    def save_economics(self) -> None:
        """Save economic state"""
        self.atp_pool.save_state()

    def load_economics(self) -> None:
        """Load economic state"""
        self.atp_pool.load_state()


# Example usage
if __name__ == "__main__":
    print("SAGE ATP/ADP Integration - Society 4 Economic Framework")
    print("=" * 60)

    # This would normally wrap Genesis's SAGE model
    # from sage.core.sage_federation_v1 import SAGE, SAGEConfig
    # sage = SAGE(SAGEConfig())
    # sage_economic = SAGEWithEconomy(sage)

    # For testing, demonstrate ATP pool
    config = AtpConfig()
    pool = AtpPool(config)

    print(f"\nInitial State:")
    print(f"  ATP Balance: {pool.atp_balance}")
    print(f"  ADP Balance: {pool.adp_balance}")

    # Simulate inference
    print(f"\nSimulating SAGE inference (H-ratio: 0.8)...")
    cost = 1 + int(0.8 * config.h_level_cost) + 2  # L + H + consciousness
    pool.discharge(cost, "SAGE inference", {"h_ratio": 0.8})

    atp, adp = pool.get_balance()
    print(f"  Cost: {cost} ATP")
    print(f"  ATP Balance: {atp}")
    print(f"  ADP Balance: {adp}")

    # Simulate reward
    print(f"\nRewarding excellent reasoning...")
    refund = int(cost * config.excellent_refund)
    pool.refund(refund, "Excellent reasoning (score: 0.9)")

    atp, adp = pool.get_balance()
    print(f"  Refund: {refund} ATP")
    print(f"  ATP Balance: {atp}")
    print(f"  ADP Balance: {adp}")

    # Show statistics
    print(f"\nStatistics:")
    stats = pool.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
