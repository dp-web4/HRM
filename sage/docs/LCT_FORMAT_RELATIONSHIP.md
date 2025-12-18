# LCT Format Relationship: URI References vs. Full Certificates

**Date**: 2025-12-17 (Session 63+)
**Author**: Legion (Autonomous Research)
**Status**: Discovery Documentation

---

## Discovery Context

During Session 63+ autonomous research, a critical architectural insight emerged: there are **two related but distinct LCT formats** in the Web4 ecosystem:

1. **LCT Core Specification** (Web4 Standard) - Full identity certificates
2. **Unified LCT Identity Specification** (Cross-System Integration) - Lightweight URI references

This document clarifies their relationship and usage patterns.

---

## The Two LCT Formats

### Format 1: LCT Core Specification (Full Certificate)

**Location**: `/web4/web4-standard/core-spec/LCT-linked-context-token.md`
**Version**: 1.0.0 (October 2025)
**Purpose**: Complete unforgeable identity certificate with full context

**Structure**:
```json
{
  "lct_id": "lct:web4:mb32:...",
  "subject": "did:web4:key:z6Mk...",
  "binding": {
    "entity_type": "human|ai|organization|role|...",
    "public_key": "mb64:coseKey",
    "hardware_anchor": "eat:mb64:hw:...",
    "created_at": "ISO 8601",
    "binding_proof": "cose:Sig_structure"
  },
  "birth_certificate": {
    "issuing_society": "lct:web4:society:...",
    "citizen_role": "lct:web4:role:citizen:...",
    "birth_witnesses": ["lct:...", "lct:...", "lct:..."]
  },
  "mrh": {
    "bound": [...],
    "paired": [...],
    "witnessing": [...],
    "horizon_depth": 3
  },
  "t3_tensor": {
    "dimensions": {
      "technical_competence": 0.85,
      "social_reliability": 0.92,
      ...
    },
    "composite_score": 0.84
  },
  "v3_tensor": {
    "dimensions": {
      "energy_balance": 140,
      "contribution_history": 0.89,
      ...
    }
  }
}
```

**Size**: ~500-2000+ lines (depending on MRH size and attestations)
**Storage**: Blockchain (ACT), distributed registry, or local certificate store
**Use Case**: Authoritative identity document with full trust/value context

---

### Format 2: Unified LCT Identity Specification (URI Reference)

**Location**: `/web4/docs/LCT_UNIFIED_IDENTITY_SPECIFICATION.md`
**Version**: 1.0.0 (December 2025)
**Purpose**: Lightweight cross-system identity reference

**Structure**:
```
lct://{component}:{instance}:{role}@{network}?{query_params}#{fragment}
```

**Examples**:
```
lct://sage:thinker:expert_42@testnet
lct://web4-agent:guardian:coordinator@mainnet?pairing_status=active&trust_threshold=0.75
lct://act-validator:node1:consensus@testnet#did:key:z6MkhaXg...
```

**Size**: ~50-200 characters (single line)
**Storage**: In-memory, configuration files, database keys, message headers
**Use Case**: Reference to full LCT certificate, cross-system coordination

---

## Relationship Between Formats

### Analogy

Think of the relationship like **URLs vs. Web Pages**:

- **URI Reference** (`lct://sage:thinker:expert_42@testnet`) = URL
- **Full Certificate** (JSON with MRH, tensors, etc.) = Full web page content

The URI is a **compact reference** that can be **resolved** to the full certificate.

### Resolution Process

```
1. SAGE component holds URI: lct://sage:thinker:expert_42@testnet
2. When full identity needed, resolve URI:
   a. Extract component (sage), instance (thinker), role (expert_42), network (testnet)
   b. Query LCT registry for network "testnet"
   c. Lookup full certificate by component:instance:role key
   d. Return full LCT certificate with MRH, T3/V3 tensors, etc.
3. Use full certificate for:
   - Trust evaluation (check T3 tensor)
   - Capability verification (check policy section)
   - Relationship validation (check MRH paired/bound)
   - Birth certificate validation (check issuing society)
```

### Concrete Example

**SAGE Expert 42 Registration**:

```python
# Step 1: Generate Unified LCT URI (lightweight reference)
from lct_identity import sage_expert_to_lct

uri = sage_expert_to_lct(
    expert_id=42,
    instance="thinker",
    network="testnet"
)
# Result: "lct://sage:thinker:expert_42@testnet"

# Step 2: Register with full LCT certificate (on ACT blockchain)
full_lct = {
    "lct_id": "lct:web4:sage:thinker:expert_42:mb32hash",
    "subject": "did:web4:key:z6Mk...",  # Expert's DID
    "binding": {
        "entity_type": "ai",
        "public_key": expert_public_key,
        "created_at": "2025-12-17T18:00:00Z",
        "binding_proof": expert_signature
    },
    "birth_certificate": {
        "issuing_society": "lct:web4:society:sage-collective",
        "citizen_role": "lct:web4:role:expert",
        "birth_witnesses": [
            "lct://sage:thinker:coordinator@testnet",
            "lct://sage:thinker:expert_1@testnet",
            "lct://sage:thinker:expert_5@testnet"
        ]
    },
    "mrh": {
        "bound": [],
        "paired": [
            {
                "lct_id": "lct://sage:thinker:coordinator@testnet",
                "pairing_type": "operational",
                "permanent": false,
                "ts": "2025-12-17T18:00:00Z"
            }
        ],
        "witnessing": [],
        "horizon_depth": 1
    },
    "t3_tensor": {
        "dimensions": {
            "technical_competence": 0.50,  # New expert
            "social_reliability": 0.60,
            "temporal_consistency": 0.40,
            "witness_count": 0.30,
            "lineage_depth": 0.10,
            "context_alignment": 0.70
        },
        "composite_score": 0.43
    },
    "v3_tensor": {
        "dimensions": {
            "energy_balance": 100,  # Initial ATP
            "contribution_history": 0.0,
            "resource_stewardship": 0.5,
            "network_effects": 0.0,
            "reputation_capital": 0.0,
            "temporal_value": 0.0
        },
        "composite_score": 0.08
    }
}

# Step 3: Store mapping (URI → Full Certificate)
registry.register(uri, full_lct)

# Step 4: Use URI for fast lookups
expert_uri = "lct://sage:thinker:expert_42@testnet"
expert_id = lct_to_sage_expert(expert_uri)  # Returns: 42

# Step 5: Resolve to full certificate when needed
full_cert = registry.resolve(expert_uri)
trust_score = full_cert["t3_tensor"]["composite_score"]  # 0.43
```

---

## ACT Blockchain LCT Structure

The ACT blockchain has its own protobuf LCT structure (simplified from full spec):

**Location**: `/act/implementation/ledger/proto/act/lctmanager/v1/lct.proto`

```protobuf
message LCT {
  string id = 1;                          // Unique ID
  string entity_type = 2;                 // human, ai, role, society, dictionary
  LCTIdentity identity = 3;               // Cryptographic keys
  MRH mrh = 4;                            // Markov Relevancy Horizon
  BirthCertificate birth_certificate = 5; // Optional
  google.protobuf.Timestamp created_at = 6;
  google.protobuf.Timestamp updated_at = 7;
  string owner = 8;                       // Cosmos address
  LCTStatus status = 9;                   // ACTIVE/SUSPENDED/REVOKED/EXPIRED
}
```

**Relationship to Unified URI**:
- The `id` field can store the Unified LCT URI: `lct://sage:thinker:expert_42@testnet`
- The full protobuf message is the on-chain representation
- Query by URI returns the full protobuf LCT

---

## Usage Patterns

### When to Use URI Format

✅ **Use Unified LCT URI when**:
- Passing identity references in function calls
- Storing expert ID mappings in memory
- Configuration files and environment variables
- Message headers for cross-system communication
- Database keys and indexes
- Logging and debugging output
- Quick identity comparisons (string equality)

**Example**:
```python
# Good: Lightweight URI reference
expert_identity = "lct://sage:thinker:expert_42@testnet"
coordinator.register_expert(expert_identity)
```

### When to Use Full Certificate

✅ **Use Full LCT Certificate when**:
- Validating trust scores before operations
- Checking capabilities and permissions
- Verifying birth certificate and witnesses
- Evaluating MRH relationships
- Computing ATP/ADP energy costs
- Blockchain transactions and state updates
- Forensic audit trails

**Example**:
```python
# Good: Full certificate for trust evaluation
full_lct = registry.resolve("lct://sage:thinker:expert_42@testnet")
if full_lct["t3_tensor"]["composite_score"] >= 0.75:
    authorize_operation(full_lct)
```

---

## Implementation Status

### ✅ Completed (Session 63+)

1. **Unified LCT URI Parsing Library** (`/HRM/sage/web4/lct_identity.py`)
   - LCTIdentity dataclass
   - parse_lct_uri(), construct_lct_uri()
   - SAGE expert conversion utilities
   - Legacy migration support
   - 32/32 unit tests passing

2. **Enhanced SAGE Expert Bridge** (`/HRM/sage/web4/expert_identity_enhanced.py`)
   - EnhancedExpertIdentityBridge class
   - Bidirectional mappings (expert ID ↔ URI)
   - Query parameter support (trust_threshold, capabilities, pairing_status)
   - Backward compatibility with legacy format

### ⏳ Pending

1. **Full LCT Certificate Generation for SAGE Experts**
   - Generate full Web4-compliant LCT certificates for SAGE experts
   - Birth certificate with SAGE coordinator as issuing society
   - T3 tensor initialization based on expert performance
   - V3 tensor with ATP allocation

2. **LCT Registry with Resolution**
   - URI → Full Certificate resolution service
   - Blockchain integration (ACT LCTManager)
   - Caching layer for fast lookups
   - Synchronization across systems

3. **Cross-System Integration Tests**
   - SAGE → ACT blockchain registration
   - Trust tensor synchronization
   - MRH relationship updates
   - Witness attestation flows

---

## Architectural Implications

### Benefits of Two-Tier Approach

1. **Performance**:
   - URI references are ~100x smaller than full certificates
   - Fast string comparisons for identity checks
   - Minimal serialization overhead

2. **Flexibility**:
   - URI format can reference certificates on any network
   - Query parameters allow context-specific metadata
   - Fragment can link to public key or DID

3. **Scalability**:
   - URIs can be cached in-memory
   - Full certificates only loaded when needed
   - Reduces blockchain query load

4. **Interoperability**:
   - URI format works across SAGE, ACT, Web4
   - Language-agnostic (Python, Go, TypeScript)
   - No dependency on full certificate schema

### Design Patterns

**Pattern 1: Lazy Resolution**
```python
class ExpertIdentity:
    def __init__(self, uri: str):
        self.uri = uri  # Lightweight reference
        self._full_cert = None  # Lazy-loaded

    @property
    def full_certificate(self):
        if self._full_cert is None:
            self._full_cert = registry.resolve(self.uri)
        return self._full_cert

    @property
    def trust_score(self):
        return self.full_certificate["t3_tensor"]["composite_score"]
```

**Pattern 2: URI as Primary Key**
```python
# Database schema
CREATE TABLE expert_identities (
    lct_uri VARCHAR(256) PRIMARY KEY,
    expert_id INTEGER NOT NULL,
    full_certificate JSONB,
    last_updated TIMESTAMP
);

# Fast lookups by URI
SELECT expert_id FROM expert_identities
WHERE lct_uri = 'lct://sage:thinker:expert_42@testnet';
```

**Pattern 3: Query Parameter Overrides**
```python
# Base URI without context
base_uri = "lct://sage:thinker:expert_42@testnet"

# Add context-specific metadata via query params
trusted_uri = f"{base_uri}?trust_threshold=0.85&pairing_status=active"
experimental_uri = f"{base_uri}?trust_threshold=0.50&pairing_status=pending"

# Parse to get context
lct = parse_lct_uri(trusted_uri)
if lct.trust_threshold >= 0.85:
    use_in_production(lct)
```

---

## Next Steps

1. **Implement Full Certificate Generator**
   - Create `sage/web4/lct_certificate_generator.py`
   - Generate Web4-compliant certificates for SAGE experts
   - Include birth certificate, MRH, T3/V3 tensors

2. **Create LCT Resolution Service**
   - Implement `sage/web4/lct_resolver.py`
   - URI → Full Certificate resolution
   - Integration with ACT blockchain LCTManager
   - Local caching layer

3. **Integration Testing**
   - End-to-end test: SAGE expert → ACT blockchain → Trust evaluation
   - Cross-system identity lookup tests
   - Performance benchmarks (URI vs. full certificate)

4. **Documentation**
   - Update SAGE documentation with LCT integration guide
   - Create developer guide for LCT URI usage
   - Document resolution protocol

---

## References

- **LCT Core Specification**: `/web4/web4-standard/core-spec/LCT-linked-context-token.md`
- **Unified LCT Identity Spec**: `/web4/docs/LCT_UNIFIED_IDENTITY_SPECIFICATION.md`
- **ACT LCT Protobuf**: `/act/implementation/ledger/proto/act/lctmanager/v1/lct.proto`
- **Session 62+ Summary**: `/private-context/moments/2025-12-17-legion-session62-cross-project-integration-architecture.md`
- **Implementation**: `/HRM/sage/web4/lct_identity.py`
- **Tests**: `/HRM/sage/tests/test_lct_identity.py`

---

**Key Insight**: The Unified LCT URI is not a replacement for the full LCT certificate—it's a **lightweight reference format** that enables cross-system identity management while maintaining compatibility with the full Web4 LCT specification. Think "URL to identity" not "identity itself."
