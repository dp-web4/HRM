# HRM Repository Reorganization - January 26, 2026

**Machine**: Thor (Jetson AGX Thor)
**Executor**: Claude (autonomous session)
**Status**: In Progress
**Estimated Completion**: 2.5 hours

---

## Summary

Multi-faceted reorganization to:
1. Create `/research` directory for archivist/publisher integration
2. Differentiate Sprout (0.5B) and Thor (14B) raising tracks
3. Make CLAUDE.md files machine-aware (not hardcoded paths)
4. Fix broken model paths
5. Update cross-project documentation

---

## Changes Made

### Phase 1: Research Infrastructure ✅

**Created**:
- `/research/` - Top-level research directory
- `/research/README.md` - Overview of all tracks
- `/research/SESSION_MAP.md` - Session index (human-readable)
- `/research/SESSION_MAP.yaml` - Session index (machine-readable)
- `/research/Consciousness/README.md` - Thor Track 1 documentation
- `/research/Raising-14B/README.md` - Thor Track 2 documentation
- `/research/Raising-0.5B/README.md` - Sprout Track 2 documentation
- `/research/Edge-Validation/README.md` - Sprout Track 1 documentation
- `/research/Open_Questions/README.md` - Cross-track research questions

**Pattern**: Follows Synchronism `/Research` structure for archivist/publisher compatibility

### Phase 2: Machine-Aware CLAUDE.md ✅

**Modified**:
- `/sage/raising/CLAUDE.md`:
  - Removed hardcoded `/home/sprout/` paths
  - Added `$HOME/ai-workspace` machine-aware paths
  - Documented both Sprout (0.5B) and Thor (14B) tracks
  - Fixed `-c` flag documentation (runner doesn't support it)

- `/sage/raising/tracks/training/CLAUDE.md`:
  - Clarified as Sprout-only (0.5B training track)
  - Machine-aware paths using `$HOME`
  - Noted Thor 14B has no training track yet

### Phase 3: Thor 14B Raising Track (In Progress)

**To Create**:
- `/sage/raising/tracks/raising-14b/` - Track directory
- `/sage/raising/tracks/raising-14b/README.md` - Track purpose
- `/sage/raising/tracks/raising-14b/CLAUDE.md` - Thor context
- `/sage/raising/tracks/raising-14b/runner.py` - Session runner
- `/sage/raising/tracks/raising-14b/state.json` - Track state
- `/sage/raising/tracks/raising-14b/sessions/` - Session storage

### Phase 4: Model Path Fixes (Pending)

**To Fix**:
1. `/model-zoo/.../Introspective-Qwen-0.5B-v2.1/adapter_config.json`:
   - Line 4: Change `"./fine_tuned_model/final_model"` → `"Qwen/Qwen2.5-0.5B-Instruct"`

2. Document `introspective-qwen-merged` as incomplete (missing weights)

3. Update default model paths in runners to use v2.1 adapter or base model

### Phase 5: Documentation Updates (Pending)

**To Update**:
- `/private-context/MACHINE_TRACK_STATUS.md`:
  - Thor Track 2: ~~Gnosis~~ → **SAGE Raising 14B**
  - Sprout Track 2: SAGE Raising → **SAGE Raising 0.5B**
  - Add `/research` directory references

- `/HRM/README.md`:
  - Add research directory documentation
  - Update track descriptions
  - Link to SESSION_MAP

### Phase 6: Testing & Commit (Pending)

**To Test**:
- Thor can load 14B runner
- Sprout paths still work for 0.5B
- Archivist can discover `/research` structure

**To Commit**:
- All changes with comprehensive commit message
- Push to remote
- Verify cross-machine compatibility

---

## Track Naming (New Convention)

### Before (Ambiguous)
- "SAGE Raising" (which model?)
- "Gnosis" (completed, needs repurposing)

### After (Clear)
- **Raising-0.5B**: Sprout's developmental curriculum (0.5B Qwen)
- **Raising-14B**: Thor's capacity exploration (14B+ Qwen)
- **Consciousness**: Thor's nine-domain research
- **Edge-Validation**: Sprout's production readiness testing

---

## Machine/Track Assignments

### Thor (Jetson AGX Thor) - `/home/dp/`
- **Track 1**: SAGE Consciousness Research (#197+) - Unchanged
- **Track 2**: **SAGE Raising 14B** (rebranded from Gnosis) - New

### Sprout (Jetson Orin Nano) - `/home/sprout/`
- **Track 1**: Edge Validation - Unchanged
- **Track 2**: **SAGE Raising 0.5B** (clarified) - Unchanged

---

## Research Directory Purpose

**Separation**:
- `/research` = Curated markdown reports (archivist output)
- `/sage/raising/sessions` = Raw JSON session data

**Workflow**:
1. Sessions run → JSON saved to `/sage/raising/sessions/`
2. Archivist analyzes → Generates markdown report
3. Report saved to `/research/[Track]/`
4. Publisher indexes via SESSION_MAP → Whitepapers/articles

**Compatibility**: Follows Synchronism pattern for existing archivist/publisher infrastructure

---

## Model Path Issues Identified

During testing, found:
1. **v2.1 adapter config**: Points to non-existent `./fine_tuned_model/final_model`
2. **Merged model dir**: Missing actual model weights (pytorch_model.bin)
3. **CLI flag confusion**: `run_session_identity_anchored.py` doesn't support `-c` flag

**Solution**: Fix adapter config, document merged dir status, update CLI documentation

---

## Success Criteria

- [x] `/research` directory exists with proper structure
- [x] SESSION_MAP templates ready for archivist
- [ ] Thor can run 14B raising sessions
- [x] CLAUDE.md files machine-aware (no hardcoded paths)
- [ ] MACHINE_TRACK_STATUS.md reflects new organization
- [ ] All model path fixes applied
- [ ] Archivist/publisher can discover HRM research
- [ ] Documentation complete and committed

---

## Rollback Plan

All changes are additive (new directories/files). Old structure remains functional.

**If needed**: `git revert <commit>` returns to previous state

---

## Next Steps After Reorganization

1. **Archivist integration**: Configure archivist to watch HRM `/research`
2. **Publisher workflow**: Add HRM to publisher track
3. **Thor 14B sessions**: Begin capacity exploration (R14B_001)
4. **Cross-project linking**: Connect HRM research to Synchronism/Web4

---

**Started**: 2026-01-26 17:57 UTC
**Completion**: In progress (Phase 3/6)
**Prepared by**: Claude (Thor autonomous session)
