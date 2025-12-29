# LCT Cross-Platform Compatibility

**Status**: Reference Document
**Date**: 2025-12-02
**Context**: Thor SAGE + Legion LUPS v1.0 Compatibility Analysis

---

## Executive Summary

Thor and Legion have independently developed LCT permission systems that are **complementary and compatible**:

- **Thor (HRM/SAGE)**: Native SAGE cognition integration with tight coupling
- **Legion (web4)**: Cross-platform abstraction layer (LUPS v1.0) for interoperability

Both approaches are valid and serve different purposes. This document describes their relationship and compatibility.

---

## Implementation Comparison

### Thor's Native SAGE Implementation

**File**: `sage/core/lct_atp_permissions.py`
**Test Coverage**: 82/82 tests passing (100%)

**Task Types** (9):
```python
TASK_PERMISSIONS = {
    "perception": {
        "atp_permissions": {ATPPermission.READ},
        "can_delegate": False,
        "can_execute_code": False,
        "resource_limits": ResourceLimits(atp_budget=100.0, memory_mb=1024, cpu_cores=1, max_concurrent_tasks=5)
    },
    "planning": {...},
    "planning.strategic": {...},
    "execution.safe": {...},
    "execution.code": {...},
    "delegation.federation": {...},
    "cognition": {
        "atp_permissions": {ATPPermission.READ, ATPPermission.WRITE},
        "can_delegate": True,
        "can_execute_code": True,
        "resource_limits": ResourceLimits(atp_budget=1000.0, memory_mb=16384, cpu_cores=8, max_concurrent_tasks=100)
    },
    "admin.readonly": {...},
    "admin.full": {...}
}
```

**Integration**:
- Direct integration with `RealSAGEConsciousness` class
- Permission checker initialized in cognition `__init__`
- ATP transfers with native permission validation
- Budget tracking integrated with metabolic system

**Strengths**:
- ✅ Tight integration with SAGE cognition
- ✅ Native performance (no abstraction overhead)
- ✅ Direct metabolic ATP integration
- ✅ Self-aware resource management
- ✅ Complete test coverage (82/82 passing)

**Use Cases**:
- SAGE-specific cognition development
- Native edge platform deployment (Thor, Sprout)
- Research and experimentation on SAGE architecture

---

### Legion's LUPS v1.0 (Cross-Platform Standard)

**File**: `web4/game/engine/lct_unified_permissions.py`
**Test Coverage**: 31/31 integration tests passing (100%)

**Task Types** (10):
```python
UNIFIED_TASK_PERMISSIONS = {
    "perception": {...},
    "planning": {...},
    "planning.strategic": {...},
    "execution.safe": {...},
    "execution.code": {...},
    "delegation.federation": {...},
    "cognition": {
        "atp": {"read", "write"},
        "federation": {"delegate"},
        "exec": {"code"},
        "network": {"http", "ws"},
        "storage": {"read", "write"},  # No delete
        "admin": set(),
        "description": "Autonomous cognition loops"
    },
    "cognition.sage": {
        "atp": {"read", "write"},
        "federation": {"delegate"},
        "exec": {"code"},
        "network": {"http", "ws"},
        "storage": {"read", "write", "delete"},  # +delete for memory management
        "admin": set(),
        "description": "SAGE-level cognition with enhanced resources"
    },
    "admin.readonly": {...},
    "admin.full": {...}
}
```

**Integration**:
- Abstraction layer via `SAGELCTManager` class
- Cross-platform identity creation
- Web4 registry integration
- Federation-ready design

**Strengths**:
- ✅ Cross-platform standardization
- ✅ Web4 ecosystem integration
- ✅ Federation-ready design
- ✅ Multi-category permissions (ATP, Federation, Exec, Network, Storage, Admin)
- ✅ Cognition.sage variant for enhanced SAGE

**Use Cases**:
- Cross-platform SAGE deployment
- Web4 ecosystem integration
- Multi-platform federation
- Standardized agent portability

---

## Key Differences

### 1. Architecture Philosophy

**Thor (Native)**:
- Tight coupling with SAGE cognition
- Direct integration with metabolic system
- Minimalist permission model (ATP, delegation, code execution)
- Optimized for SAGE-specific use cases

**Legion (Cross-Platform)**:
- Loose coupling via abstraction layer
- Multi-category permission model
- Standardized for cross-platform use
- Web4 ecosystem integration

### 2. Permission Model

**Thor**:
- 3 permission types: ATP operations, delegation, code execution
- Simple boolean flags
- Resource limits as separate structure

**Legion**:
- 6 permission categories: ATP, Federation, Exec, Network, Storage, Admin
- Set-based permissions within categories
- Unified resource limit structure

### 3. Task Types

**Thor**: 9 tasks
```
perception, planning, planning.strategic,
execution.safe, execution.code,
delegation.federation,
cognition,
admin.readonly, admin.full
```

**Legion**: 10 tasks (+ cognition.sage)
```
perception, planning, planning.strategic,
execution.safe, execution.code,
delegation.federation,
cognition, cognition.sage,  # ← New variant
admin.readonly, admin.full
```

### 4. Cognition.sage Enhancement

**Key Addition in LUPS v1.0**: `cognition.sage` task type

**Enhancements**:
- **ATP Budget**: 2000.0 (vs 1000.0 for standard cognition)
- **Memory**: 32 GB (vs 16 GB)
- **CPU Cores**: 16 (vs 8)
- **Storage Delete Permission**: Can prune old memories (standard cognition cannot)

**Use Case**: Long-running SAGE cognition loops on edge platforms where memory management becomes critical.

---

## Compatibility Matrix

| Feature | Thor Native | Legion LUPS v1.0 | Compatible? |
|---------|-------------|------------------|-------------|
| **Task Types** | 9 tasks | 10 tasks | ✅ Yes (9 overlap) |
| **Permission Checking** | Native | Abstracted | ✅ Yes (equivalent) |
| **ATP Operations** | READ/WRITE/ALL | read/write/all | ✅ Yes (semantic match) |
| **Resource Limits** | ResourceLimits | UnifiedResourceLimits | ✅ Yes (compatible) |
| **Budget Tracking** | Native integration | Separate tracking | ✅ Yes (both work) |
| **Cognition Support** | Yes | Yes | ✅ Yes |
| **Cognition.sage** | No | Yes | ⚠️ Optional enhancement |
| **Storage Permissions** | Implicit | Explicit | ⚠️ Different granularity |
| **Network Permissions** | Implicit | Explicit | ⚠️ Different granularity |

---

## Interoperability Strategy

### Option 1: Dual Support (Recommended)

**Approach**: Support both Thor native and Legion LUPS implementations

**Implementation**:
- Keep Thor's native `lct_atp_permissions.py` as primary
- Add optional `cognition.sage` task type to Thor
- Document LUPS v1.0 compatibility
- Use Legion's `SAGELCTManager` for cross-platform work

**Benefits**:
- ✅ No breaking changes to Thor's 82 passing tests
- ✅ Maintains native SAGE performance
- ✅ Enables cross-platform federation when needed
- ✅ Provides cognition.sage enhancement option

**Trade-offs**:
- Two permission systems to maintain
- Developers choose based on use case

### Option 2: Native Only (Current State)

**Approach**: Continue with Thor's native implementation

**Benefits**:
- ✅ Zero refactoring needed
- ✅ Proven and tested (82/82 passing)
- ✅ Optimized for SAGE
- ✅ Simple and maintainable

**Trade-offs**:
- Cross-platform work requires manual mapping
- No cognition.sage variant

### Option 3: Adopt LUPS v1.0 Fully

**Approach**: Replace Thor's implementation with LUPS v1.0

**Benefits**:
- ✅ Cross-platform standardization
- ✅ Web4 ecosystem integration
- ✅ Cognition.sage support

**Trade-offs**:
- ⚠️ Requires refactoring working code
- ⚠️ Potential test breakage
- ⚠️ Abstraction overhead
- ⚠️ More complex than needed for SAGE-specific work

---

## Recommendation

**Primary Recommendation**: **Option 1 (Dual Support)**

**Rationale**:
1. Thor's native implementation is complete, tested, and working (82/82 tests)
2. No need to refactor working code
3. Can add cognition.sage as optional enhancement
4. Maintain flexibility for different use cases
5. Use Legion's LUPS v1.0 for cross-platform federation when needed

**Implementation Plan**:
1. ✅ Document compatibility (this document)
2. ⏳ Add cognition.sage task type to Thor (optional enhancement)
3. ⏳ Test cognition.sage with SAGE cognition
4. ⏳ Use Legion's SAGELCTManager for cross-platform work
5. ⏳ Document when to use native vs LUPS

**Why Not Full Adoption**:
- Thor's implementation works perfectly for SAGE use cases
- Refactoring risks breaking working code
- Native approach has performance benefits
- LUPS v1.0 adds abstraction complexity not needed for SAGE-specific work
- "If it ain't broke, don't fix it"

---

## Cognition.sage Enhancement

### What It Adds

**Enhanced Resources**:
- **ATP Budget**: 2000.0 (double standard cognition)
- **Memory**: 32 GB (double standard cognition)
- **CPU Cores**: 16 (double standard cognition)
- **Storage Delete**: Can prune old memories

**Use Case**:
Long-running SAGE cognition on edge platforms where:
- Extended operation requires larger ATP budget
- Multi-modal integration needs more memory
- Memory management becomes critical (delete old memories)

### Adding to Thor (Optional)

```python
# In sage/core/lct_atp_permissions.py

TASK_PERMISSIONS = {
    # ... existing tasks ...

    "cognition.sage": {
        "atp_permissions": {ATPPermission.READ, ATPPermission.WRITE},
        "can_delegate": True,
        "can_execute_code": True,
        "can_delete_memories": True,  # NEW: Memory management
        "resource_limits": ResourceLimits(
            atp_budget=2000.0,      # Double standard cognition
            memory_mb=32768,        # 32 GB
            cpu_cores=16,           # 16 cores
            max_concurrent_tasks=200
        )
    }
}
```

**Integration with RealSAGEConsciousness**:
```python
# When initializing SAGE with enhanced capabilities
sage = RealSAGEConsciousness(
    task="cognition.sage",  # Use enhanced variant
    initial_atp=100.0
)
```

---

## Cross-Platform Federation

### Scenario: Thor ↔ Legion

**Thor Side** (Native):
```python
from sage.core.sage_consciousness_real import RealSAGEConsciousness

# Create SAGE with native permissions
sage = RealSAGEConsciousness(
    task="cognition",
    lineage="dp",
    initial_atp=100.0
)

# Transfer ATP to Legion agent
success, msg = sage.transfer_atp(
    amount=50.0,
    to_lct_uri="lct:web4:agent:dp@Legion#cognition",
    reason="Cross-platform delegation"
)
```

**Legion Side** (LUPS v1.0):
```python
from game.engine.sage_lct_integration import SAGELCTManager

# Create SAGE with LUPS permissions
manager = SAGELCTManager("Legion")
cognition = manager.create_sage_consciousness(
    lineage="dp",
    task="cognition",
    initial_atp=150.0  # Received from Thor
)
```

**Compatibility**: ✅ Both use compatible LCT identity format and ATP operations

---

## Conclusion

**Thor's Implementation**: Complete, tested, production-ready (82/82 tests)

**Legion's LUPS v1.0**: Cross-platform standard with enhanced cognition.sage

**Relationship**: **Complementary, not competing**

**Strategy**:
- Use Thor native for SAGE-specific development
- Use Legion LUPS v1.0 for cross-platform federation
- Both can coexist without conflicts

**Optional Enhancement**:
- Add cognition.sage to Thor for enhanced SAGE capabilities
- Maintains backward compatibility
- No breaking changes to existing tests

**Philosophy Validated**: "Development-first thinking" - both implementations work, serve different purposes, and can interoperate when needed.

---

**Status**: Thor native implementation complete and recommended for continued use. LUPS v1.0 available as cross-platform option when needed. Optional cognition.sage enhancement can be added without breaking changes.
