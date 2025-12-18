#!/usr/bin/env python3
"""
LCT Resolver - URI to Full Certificate Resolution

Provides lightweight URI → Full Certificate resolution with caching,
registry support, and integration with ACT blockchain.

Created: Session 63+ (2025-12-17)
Author: Legion (Autonomous Research)
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass
from collections import OrderedDict

from sage.web4.lct_identity import (
    parse_lct_uri,
    validate_lct_uri,
    LCTIdentity
)


@dataclass
class ResolutionResult:
    """Result of LCT URI resolution."""
    uri: str
    certificate: Optional[Dict[str, Any]]
    source: str  # "cache", "file", "blockchain", "generator", "remote"
    latency_ms: float
    success: bool
    error: Optional[str] = None


class LRUCache:
    """Simple LRU cache for LCT certificates."""

    def __init__(self, max_size: int = 100):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of certificates to cache
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def get(self, uri: str) -> Optional[Dict[str, Any]]:
        """Get certificate from cache, updating LRU order."""
        if uri not in self.cache:
            return None

        # Move to end (most recently used)
        self.cache.move_to_end(uri)
        return self.cache[uri]

    def put(self, uri: str, certificate: Dict[str, Any]) -> None:
        """Put certificate in cache, evicting LRU if needed."""
        if uri in self.cache:
            # Update existing entry
            self.cache.move_to_end(uri)
            self.cache[uri] = certificate
        else:
            # Add new entry
            self.cache[uri] = certificate

            # Evict LRU if cache is full
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)  # Remove oldest

    def clear(self) -> None:
        """Clear all cached certificates."""
        self.cache.clear()

    def __len__(self) -> int:
        """Return number of cached certificates."""
        return len(self.cache)


class LCTResolver:
    """
    LCT URI to Full Certificate resolver.

    Provides multi-tier resolution strategy:
    1. In-memory cache (LRU)
    2. Local file registry
    3. Generator callback (for dynamic generation)
    4. Blockchain query (future)
    5. Remote registry (future)
    """

    def __init__(
        self,
        cache_size: int = 100,
        registry_dir: Optional[Path] = None,
        enable_cache: bool = True
    ):
        """
        Initialize LCT resolver.

        Args:
            cache_size: Maximum number of certificates to cache
            registry_dir: Optional directory for local certificate storage
            enable_cache: Enable in-memory caching
        """
        self.cache = LRUCache(max_size=cache_size) if enable_cache else None
        self.registry_dir = Path(registry_dir) if registry_dir else None
        self.enable_cache = enable_cache

        # Pluggable resolvers (priority order)
        self.resolvers: Dict[str, Callable[[str], Optional[Dict[str, Any]]]] = {}

        # Statistics
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "file_hits": 0,
            "generator_hits": 0,
            "blockchain_hits": 0,
            "remote_hits": 0,
            "failures": 0
        }

    def register_resolver(self, name: str, resolver_fn: Callable[[str], Optional[Dict[str, Any]]]) -> None:
        """
        Register a custom resolver function.

        Args:
            name: Resolver name (e.g., "blockchain", "generator")
            resolver_fn: Function that takes URI and returns certificate dict or None
        """
        self.resolvers[name] = resolver_fn

    def resolve(self, uri: str, skip_cache: bool = False) -> ResolutionResult:
        """
        Resolve LCT URI to full certificate.

        Args:
            uri: LCT URI to resolve
            skip_cache: Skip cache and force fresh resolution

        Returns:
            Resolution result with certificate and metadata
        """
        start_time = time.time()

        # Validate URI format
        if not validate_lct_uri(uri):
            return ResolutionResult(
                uri=uri,
                certificate=None,
                source="validation",
                latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error=f"Invalid LCT URI format: {uri}"
            )

        # Try cache first (if enabled and not skipped)
        if self.enable_cache and not skip_cache and self.cache:
            cached = self.cache.get(uri)
            if cached is not None:
                self.stats["cache_hits"] += 1
                return ResolutionResult(
                    uri=uri,
                    certificate=cached,
                    source="cache",
                    latency_ms=(time.time() - start_time) * 1000,
                    success=True
                )
            self.stats["cache_misses"] += 1

        # Try file registry
        if self.registry_dir:
            file_cert = self._resolve_from_file(uri)
            if file_cert is not None:
                self.stats["file_hits"] += 1
                if self.cache:
                    self.cache.put(uri, file_cert)
                return ResolutionResult(
                    uri=uri,
                    certificate=file_cert,
                    source="file",
                    latency_ms=(time.time() - start_time) * 1000,
                    success=True
                )

        # Try registered resolvers (generator, blockchain, etc.)
        for resolver_name, resolver_fn in self.resolvers.items():
            try:
                cert = resolver_fn(uri)
                if cert is not None:
                    self.stats[f"{resolver_name}_hits"] += 1
                    if self.cache:
                        self.cache.put(uri, cert)
                    return ResolutionResult(
                        uri=uri,
                        certificate=cert,
                        source=resolver_name,
                        latency_ms=(time.time() - start_time) * 1000,
                        success=True
                    )
            except Exception as e:
                # Log error but continue to next resolver
                print(f"Resolver {resolver_name} failed for {uri}: {e}")

        # Resolution failed
        self.stats["failures"] += 1
        return ResolutionResult(
            uri=uri,
            certificate=None,
            source="none",
            latency_ms=(time.time() - start_time) * 1000,
            success=False,
            error=f"No resolver found certificate for URI: {uri}"
        )

    def _resolve_from_file(self, uri: str) -> Optional[Dict[str, Any]]:
        """
        Resolve certificate from local file registry.

        File naming convention:
        lct_cert_{instance}_{role}_{network}.json
        """
        if not self.registry_dir or not self.registry_dir.exists():
            return None

        # Parse URI to get components
        lct = parse_lct_uri(uri)

        # Try to find matching file
        # Pattern: lct_cert_{instance}_{role}_{network}.json
        filename = f"lct_cert_{lct.instance}_{lct.role}_{lct.network}.json"
        filepath = self.registry_dir / filename

        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    cert = json.load(f)
                return cert
            except Exception as e:
                print(f"Error loading certificate from {filepath}: {e}")
                return None

        return None

    def register_certificate(self, uri: str, certificate: Dict[str, Any], persist: bool = True) -> None:
        """
        Register a certificate in the resolver.

        Args:
            uri: LCT URI
            certificate: Full certificate dictionary
            persist: Whether to persist to file registry
        """
        # Add to cache
        if self.cache:
            self.cache.put(uri, certificate)

        # Persist to file if enabled
        if persist and self.registry_dir:
            self._save_to_file(uri, certificate)

    def _save_to_file(self, uri: str, certificate: Dict[str, Any]) -> None:
        """Save certificate to local file registry."""
        if not self.registry_dir:
            return

        self.registry_dir.mkdir(parents=True, exist_ok=True)

        # Parse URI to get components
        lct = parse_lct_uri(uri)

        # Save to file
        filename = f"lct_cert_{lct.instance}_{lct.role}_{lct.network}.json"
        filepath = self.registry_dir / filename

        with open(filepath, 'w') as f:
            json.dump(certificate, f, indent=2)

    def get_stats(self) -> Dict[str, Any]:
        """Get resolver statistics."""
        total_requests = sum([
            self.stats["cache_hits"],
            self.stats["cache_misses"]
        ])

        cache_hit_rate = (
            self.stats["cache_hits"] / total_requests
            if total_requests > 0 else 0.0
        )

        return {
            **self.stats,
            "cache_size": len(self.cache) if self.cache else 0,
            "total_requests": total_requests,
            "cache_hit_rate": cache_hit_rate
        }

    def clear_cache(self) -> None:
        """Clear in-memory cache."""
        if self.cache:
            self.cache.clear()

    def resolve_batch(self, uris: list[str]) -> Dict[str, ResolutionResult]:
        """
        Resolve multiple URIs in batch.

        Args:
            uris: List of LCT URIs to resolve

        Returns:
            Dictionary mapping URI to resolution result
        """
        results = {}
        for uri in uris:
            results[uri] = self.resolve(uri)
        return results


# Convenience functions
def create_resolver_with_generator(generator_callback: Callable[[str], Optional[Dict[str, Any]]],
                                   registry_dir: Optional[Path] = None,
                                   cache_size: int = 100) -> LCTResolver:
    """
    Create LCT resolver with generator callback.

    Args:
        generator_callback: Function to generate certificates on-demand
        registry_dir: Optional file registry directory
        cache_size: Cache size

    Returns:
        Configured LCT resolver
    """
    resolver = LCTResolver(
        cache_size=cache_size,
        registry_dir=registry_dir,
        enable_cache=True
    )

    resolver.register_resolver("generator", generator_callback)

    return resolver


# Example usage
if __name__ == "__main__":
    from lct_certificate_generator import SAGELCTCertificateGenerator

    print("LCT Resolver - Example Usage")
    print("=" * 60)

    # Setup
    registry_dir = Path(__file__).parent.parent / "lct_certificates"
    generator = SAGELCTCertificateGenerator(instance="thinker", network="testnet")

    # Create resolver with generator callback
    def generator_callback(uri: str) -> Optional[Dict[str, Any]]:
        """Generate certificate on-demand for SAGE experts."""
        try:
            lct = parse_lct_uri(uri)

            # Only generate for SAGE component
            if lct.component != "sage":
                return None

            # Extract expert ID if it's an expert role
            if lct.role.startswith("expert_"):
                expert_id = int(lct.role.split("_")[1])
                cert = generator.generate_expert_certificate(expert_id)
                return cert.to_dict()

            # Generate coordinator
            if lct.role == "coordinator":
                cert = generator.generate_coordinator_certificate()
                return cert.to_dict()

            return None
        except Exception as e:
            print(f"Generator failed for {uri}: {e}")
            return None

    resolver = create_resolver_with_generator(
        generator_callback=generator_callback,
        registry_dir=registry_dir,
        cache_size=50
    )

    print("\n1. Resolve from file (existing certificate)...")
    result1 = resolver.resolve("lct://sage:thinker:expert_42@testnet")
    print(f"   URI: {result1.uri}")
    print(f"   Source: {result1.source}")
    print(f"   Latency: {result1.latency_ms:.2f}ms")
    print(f"   Success: {result1.success}")
    if result1.certificate:
        print(f"   Trust Score: {result1.certificate['t3_tensor']['composite_score']}")

    print("\n2. Resolve again (should hit cache)...")
    result2 = resolver.resolve("lct://sage:thinker:expert_42@testnet")
    print(f"   Source: {result2.source}")
    print(f"   Latency: {result2.latency_ms:.2f}ms")

    print("\n3. Resolve new expert (should use generator)...")
    result3 = resolver.resolve("lct://sage:thinker:expert_99@testnet")
    print(f"   URI: {result3.uri}")
    print(f"   Source: {result3.source}")
    print(f"   Latency: {result3.latency_ms:.2f}ms")
    print(f"   Success: {result3.success}")
    if result3.certificate:
        print(f"   LCT ID: {result3.certificate['lct_id']}")
        print(f"   Trust Score: {result3.certificate['t3_tensor']['composite_score']}")

    print("\n4. Batch resolution...")
    uris = [
        "lct://sage:thinker:coordinator@testnet",
        "lct://sage:thinker:expert_1@testnet",
        "lct://sage:thinker:expert_2@testnet"
    ]
    results = resolver.resolve_batch(uris)
    for uri, result in results.items():
        print(f"   {uri}: {result.source} ({result.latency_ms:.2f}ms)")

    print("\n5. Resolver statistics...")
    stats = resolver.get_stats()
    print(f"   Cache hits: {stats['cache_hits']}")
    print(f"   Cache misses: {stats['cache_misses']}")
    print(f"   Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"   Cache size: {stats['cache_size']}")
    print(f"   File hits: {stats['file_hits']}")
    print(f"   Generator hits: {stats['generator_hits']}")
    print(f"   Failures: {stats['failures']}")

    print("\n6. Test lazy resolution pattern...")
    class ExpertIdentity:
        """Example lazy-loading identity wrapper."""
        def __init__(self, uri: str, resolver: LCTResolver):
            self.uri = uri
            self.resolver = resolver
            self._full_cert = None

        @property
        def full_certificate(self):
            """Lazy-load full certificate."""
            if self._full_cert is None:
                result = self.resolver.resolve(self.uri)
                if result.success:
                    self._full_cert = result.certificate
            return self._full_cert

        @property
        def trust_score(self):
            """Get trust score (triggers lazy load if needed)."""
            cert = self.full_certificate
            if cert:
                return cert["t3_tensor"]["composite_score"]
            return None

    expert = ExpertIdentity("lct://sage:thinker:expert_42@testnet", resolver)
    print(f"   Expert URI: {expert.uri}")
    print(f"   Trust Score (lazy-loaded): {expert.trust_score}")
    print(f"   Certificate loaded: {expert._full_cert is not None}")

    print("\n✓ LCT Resolver test complete!")
