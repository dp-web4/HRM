# HRM Archive

This directory contains completed, closed, or superseded work. **Nothing is deleted** - everything is preserved for reference and historical context.

---

## Directory Structure

| Directory | Contents | Size |
|-----------|----------|------|
| **[closed-arcs/](closed-arcs/)** | Completed research arcs with learnings documented | Variable |
| **[experiments-phase1/](experiments-phase1/)** | Early-phase experiments (Aug-Oct 2025) | ~503MB |
| **[training-experiments/](training-experiments/)** | Training run logs and results | ~854MB |
| **[legacy/](legacy/)** | Obsolete implementations and superseded docs | Variable |
| **[old-implementations/](old-implementations/)** | Previous code versions | Variable |
| **[documentation/](documentation/)** | Historical documentation | Variable |

---

## Closed Research Arcs

### ARC-AGI Track
**Status**: Closed (September 2025)
**Location**: [arc-agi/](arc-agi/)

**What We Tried**: Knowledge distillation from GR00T for ARC-AGI abstract reasoning tasks.

**What We Learned**:
- Achieved 94.45% pixel accuracy but 0% exact task matches
- Class imbalance problem: Model learned to output common patterns, not solve tasks
- **Key Insight**: SAGE is an attention orchestrator, not a task-solver. ARC-AGI requires conceptual reasoning that pattern matching can't provide.

**Lesson**: "The whole ARC-AGI test is about conceptual thinking. No amount of pattern matching is going to do it."

### Gnosis Track
**Status**: Merged into Raising-14B (January 2026)
**Location**: Gnosis materials integrated into [research/Raising-14B/](../research/Raising-14B/)

**What It Was**: Exploration of consciousness and epistemic boundaries with larger models.

**What Happened**: Renamed to "Raising-14B" to align with the Raising-0.5B track naming convention and clarify the capacity comparison focus.

---

## Large Data Directories

### experiments-phase1/ (~503MB)
Early-phase experiments from August-October 2025 including:
- Initial HRM architecture exploration
- Mock GR00T integration (before discovering real GR00T was installed)
- Early consciousness loop prototypes
- SNARC memory integration tests

**Status**: Historical record. No longer actively used but preserved for reference.

### training-experiments/ (~854MB)
Training run logs and checkpoints including:
- VAE distillation experiments (TinyVAE achievement)
- Early SAGE training attempts
- GPU mailbox performance tests
- Model comparison benchmarks

**Status**: Contains valuable baseline data for comparison. Large files may be compressed.

---

## Legacy Code and Documentation

### old-implementations/
Previous versions of core systems before major refactors:
- Pre-IRP plugin architecture
- Original orchestrator designs
- Early attention mechanisms

### legacy/
Obsolete documentation that has been superseded:
- README_OLD.md - Previous root README
- Historical reorganization plans
- Superseded status documents

---

## What to Archive vs Delete

**Archive** (move here):
- Completed research with learnings documented
- Superseded implementations with historical value
- Old documentation that might inform future decisions
- Training/experiment data that establishes baselines

**Do NOT archive** (can be deleted/gitignored):
- Build artifacts (build-logs/, can be regenerated)
- Cached dependencies (pytorch_wheels/, can be re-downloaded)
- Temporary files and test outputs

---

## Navigating the Archive

For each archived item:
1. Check the directory's README.md for context
2. Look for "LESSONS_LEARNED.md" or equivalent
3. Check the date range to understand the era
4. Cross-reference with [research/SESSION_MAP.md](../research/SESSION_MAP.md) for related sessions

---

*Archive maintained as part of HRM reorganization, February 2026*
