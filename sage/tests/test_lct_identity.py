#!/usr/bin/env python3
"""
Unit Tests for LCT Identity Parsing Library

Tests the Unified LCT Identity Specification implementation.

Created: Session 63+ (2025-12-17)
Author: Legion (Autonomous Research)
"""

import unittest
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from web4.lct_identity import (
    LCTIdentity,
    parse_lct_uri,
    validate_lct_uri,
    construct_lct_uri,
    sage_expert_to_lct,
    lct_to_sage_expert,
    migrate_legacy_expert_id,
    get_test_vectors,
)


class TestLCTIdentityParsing(unittest.TestCase):
    """Test LCT URI parsing."""

    def test_parse_simple_uri(self):
        """Test parsing simple LCT URI."""
        uri = "lct://sage:thinker:expert_42@testnet"
        lct = parse_lct_uri(uri)

        self.assertEqual(lct.component, "sage")
        self.assertEqual(lct.instance, "thinker")
        self.assertEqual(lct.role, "expert_42")
        self.assertEqual(lct.network, "testnet")
        self.assertEqual(lct.version, "1.0.0")

    def test_parse_with_query_params(self):
        """Test parsing URI with query parameters."""
        uri = "lct://sage:thinker:expert_42@testnet?trust_threshold=0.75&pairing_status=active"
        lct = parse_lct_uri(uri)

        self.assertEqual(lct.trust_threshold, 0.75)
        self.assertEqual(lct.pairing_status, "active")

    def test_parse_with_capabilities(self):
        """Test parsing URI with capabilities."""
        uri = "lct://sage:thinker:expert_42@testnet?capabilities=text-generation,reasoning"
        lct = parse_lct_uri(uri)

        self.assertIsNotNone(lct.capabilities)
        self.assertEqual(len(lct.capabilities), 2)
        self.assertIn("text-generation", lct.capabilities)
        self.assertIn("reasoning", lct.capabilities)

    def test_parse_with_fragment(self):
        """Test parsing URI with public key hash fragment."""
        uri = "lct://sage:thinker:expert_42@testnet#did:key:z6MkhaXg"
        lct = parse_lct_uri(uri)

        self.assertEqual(lct.public_key_hash, "did:key:z6MkhaXg")

    def test_parse_invalid_scheme(self):
        """Test parsing with invalid scheme raises ValueError."""
        with self.assertRaises(ValueError):
            parse_lct_uri("https://invalid.com")

    def test_parse_invalid_format(self):
        """Test parsing with invalid format raises ValueError."""
        with self.assertRaises(ValueError):
            parse_lct_uri("lct://invalid_format")


class TestLCTIdentityConstruction(unittest.TestCase):
    """Test LCT URI construction."""

    def test_construct_simple(self):
        """Test constructing simple LCT URI."""
        uri = construct_lct_uri("sage", "thinker", "expert_42", network="testnet")

        self.assertEqual(uri, "lct://sage:thinker:expert_42@testnet")

    def test_construct_with_trust_threshold(self):
        """Test constructing with trust threshold."""
        uri = construct_lct_uri(
            "sage", "thinker", "expert_42",
            network="testnet",
            trust_threshold=0.75
        )

        self.assertIn("trust_threshold=0.75", uri)

    def test_construct_with_capabilities(self):
        """Test constructing with capabilities."""
        uri = construct_lct_uri(
            "sage", "thinker", "expert_42",
            network="testnet",
            capabilities=["text-generation", "reasoning"]
        )

        lct = parse_lct_uri(uri)
        self.assertEqual(len(lct.capabilities), 2)

    def test_roundtrip(self):
        """Test parse → construct roundtrip."""
        original = "lct://sage:thinker:expert_42@testnet?trust_threshold=0.75"
        lct = parse_lct_uri(original)
        reconstructed = lct.lct_uri

        # Parse both and compare fields
        lct_original = parse_lct_uri(original)
        lct_reconstructed = parse_lct_uri(reconstructed)

        self.assertEqual(lct_original.component, lct_reconstructed.component)
        self.assertEqual(lct_original.instance, lct_reconstructed.instance)
        self.assertEqual(lct_original.role, lct_reconstructed.role)
        self.assertEqual(lct_original.network, lct_reconstructed.network)
        self.assertEqual(lct_original.trust_threshold, lct_reconstructed.trust_threshold)


class TestLCTIdentityValidation(unittest.TestCase):
    """Test LCT URI validation."""

    def test_validate_valid_uri(self):
        """Test validation of valid URI."""
        self.assertTrue(validate_lct_uri("lct://sage:thinker:expert_42@testnet"))

    def test_validate_invalid_scheme(self):
        """Test validation rejects invalid scheme."""
        self.assertFalse(validate_lct_uri("https://invalid.com"))

    def test_validate_invalid_format(self):
        """Test validation rejects invalid format."""
        self.assertFalse(validate_lct_uri("lct://invalid"))

    def test_validate_with_query_params(self):
        """Test validation accepts query parameters."""
        self.assertTrue(
            validate_lct_uri("lct://sage:thinker:expert_42@testnet?trust_threshold=0.75")
        )


class TestSAGEExpertConversion(unittest.TestCase):
    """Test SAGE expert ID conversion utilities."""

    def test_expert_to_lct(self):
        """Test converting expert ID to LCT URI."""
        uri = sage_expert_to_lct(42, instance="thinker", network="testnet")

        self.assertEqual(uri, "lct://sage:thinker:expert_42@testnet")

    def test_lct_to_expert(self):
        """Test extracting expert ID from LCT URI."""
        uri = "lct://sage:thinker:expert_42@testnet"
        expert_id = lct_to_sage_expert(uri)

        self.assertEqual(expert_id, 42)

    def test_lct_to_expert_non_sage(self):
        """Test extracting returns None for non-SAGE LCT."""
        uri = "lct://web4-agent:guardian:coordinator@mainnet"
        expert_id = lct_to_sage_expert(uri)

        self.assertIsNone(expert_id)

    def test_lct_to_expert_non_expert_role(self):
        """Test extracting returns None for non-expert role."""
        uri = "lct://sage:thinker:coordinator@testnet"
        expert_id = lct_to_sage_expert(uri)

        self.assertIsNone(expert_id)

    def test_roundtrip_expert_conversion(self):
        """Test expert ID → LCT URI → expert ID roundtrip."""
        original_id = 99
        uri = sage_expert_to_lct(original_id, instance="dreamer", network="mainnet")
        extracted_id = lct_to_sage_expert(uri)

        self.assertEqual(original_id, extracted_id)


class TestLegacyMigration(unittest.TestCase):
    """Test legacy format migration."""

    def test_migrate_simple_legacy_id(self):
        """Test migrating simple legacy expert ID."""
        legacy = "sage_thinker_expert_42"
        uri = migrate_legacy_expert_id(legacy, network="testnet")

        self.assertEqual(uri, "lct://sage:thinker:expert_42@testnet")

    def test_migrate_to_mainnet(self):
        """Test migrating with mainnet network."""
        legacy = "sage_dreamer_expert_123"
        uri = migrate_legacy_expert_id(legacy, network="mainnet")

        lct = parse_lct_uri(uri)
        self.assertEqual(lct.network, "mainnet")

    def test_migrate_invalid_format(self):
        """Test migrating invalid format raises ValueError."""
        with self.assertRaises(ValueError):
            migrate_legacy_expert_id("invalid")


class TestLCTIdentityDataclass(unittest.TestCase):
    """Test LCTIdentity dataclass."""

    def test_create_minimal(self):
        """Test creating minimal LCTIdentity."""
        lct = LCTIdentity(
            component="sage",
            instance="thinker",
            role="expert_42",
            network="testnet"
        )

        self.assertEqual(lct.component, "sage")
        self.assertEqual(lct.version, "1.0.0")  # Default

    def test_create_with_all_fields(self):
        """Test creating with all fields."""
        lct = LCTIdentity(
            component="sage",
            instance="thinker",
            role="expert_42",
            network="testnet",
            pairing_status="active",
            trust_threshold=0.75,
            capabilities=["text-generation"],
            public_key_hash="did:key:abc123"
        )

        self.assertEqual(lct.pairing_status, "active")
        self.assertEqual(lct.trust_threshold, 0.75)
        self.assertEqual(len(lct.capabilities), 1)
        self.assertEqual(lct.public_key_hash, "did:key:abc123")

    def test_lct_uri_property(self):
        """Test lct_uri property reconstruction."""
        lct = LCTIdentity(
            component="sage",
            instance="thinker",
            role="expert_42",
            network="testnet"
        )

        uri = lct.lct_uri
        self.assertTrue(uri.startswith("lct://sage:thinker:expert_42@testnet"))

    def test_invalid_component_name(self):
        """Test invalid component name raises ValueError."""
        with self.assertRaises(ValueError):
            LCTIdentity(
                component="SAGE",  # Must be lowercase
                instance="thinker",
                role="expert_42",
                network="testnet"
            )

    def test_invalid_pairing_status(self):
        """Test invalid pairing status raises ValueError."""
        with self.assertRaises(ValueError):
            LCTIdentity(
                component="sage",
                instance="thinker",
                role="expert_42",
                network="testnet",
                pairing_status="invalid_status"
            )

    def test_invalid_trust_threshold(self):
        """Test invalid trust threshold raises ValueError."""
        with self.assertRaises(ValueError):
            LCTIdentity(
                component="sage",
                instance="thinker",
                role="expert_42",
                network="testnet",
                trust_threshold=1.5  # Must be [0, 1]
            )


class TestTestVectors(unittest.TestCase):
    """Test specification test vectors."""

    def test_all_test_vectors(self):
        """Test all specification test vectors parse correctly."""
        vectors = get_test_vectors()

        for vector in vectors:
            uri = vector["uri"]
            expected = vector["parsed"]

            lct = parse_lct_uri(uri)

            for key, expected_value in expected.items():
                actual_value = getattr(lct, key)
                self.assertEqual(
                    actual_value,
                    expected_value,
                    f"Test vector {uri}: {key} mismatch"
                )


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""

    def test_empty_metadata(self):
        """Test LCT with empty metadata."""
        lct = LCTIdentity(
            component="sage",
            instance="thinker",
            role="expert_42",
            network="testnet",
            metadata={}
        )

        self.assertEqual(len(lct.metadata), 0)

    def test_custom_metadata(self):
        """Test LCT with custom metadata."""
        lct = LCTIdentity(
            component="sage",
            instance="thinker",
            role="expert_42",
            network="testnet",
            metadata={"custom_field": "value"}
        )

        uri = lct.lct_uri
        parsed = parse_lct_uri(uri)

        self.assertIn("custom_field", parsed.metadata)
        self.assertEqual(parsed.metadata["custom_field"], "value")

    def test_to_dict(self):
        """Test converting LCTIdentity to dict."""
        lct = LCTIdentity(
            component="sage",
            instance="thinker",
            role="expert_42",
            network="testnet",
            trust_threshold=0.75
        )

        data = lct.to_dict()

        self.assertIn("component", data)
        self.assertIn("trust_threshold", data)
        self.assertIn("lct_uri", data)
        self.assertEqual(data["trust_threshold"], 0.75)


def run_tests():
    """Run all tests and report results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLCTIdentityParsing))
    suite.addTests(loader.loadTestsFromTestCase(TestLCTIdentityConstruction))
    suite.addTests(loader.loadTestsFromTestCase(TestLCTIdentityValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestSAGEExpertConversion))
    suite.addTests(loader.loadTestsFromTestCase(TestLegacyMigration))
    suite.addTests(loader.loadTestsFromTestCase(TestLCTIdentityDataclass))
    suite.addTests(loader.loadTestsFromTestCase(TestTestVectors))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
