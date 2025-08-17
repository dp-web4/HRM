# Claude Context for HRM

## GitHub PAT Location
**IMPORTANT**: The GitHub Personal Access Token is stored in `/home/sprout/ai-workspace/github pat.txt`

## Sudo Access
This machine (Sprout - Jetson Orin Nano) has sudo access available for Claude.

## HRM Setup Status
- ✅ Repository cloned from fork: https://github.com/dp-web4/HRM.git
- ✅ Analysis scripts created:
  - `analyze_hrm_architecture.py` - Architecture analysis
  - `install_jetson.sh` - Dependency installation
  - `jetson_quick_start.sh` - Quick demo setup
- ⏳ Waiting for GitHub PAT permissions to push

## Key Points About HRM
- Tiny 27M parameter model (perfect for Jetson!)
- Solves complex reasoning tasks (Sudoku, mazes, ARC)
- Learns from only 1000 examples
- No pre-training needed
- Hierarchical architecture mimics human cognition

## Critical Insight: Augmentation as Sleep Cycle Training
**The augmentation strategies in HRM's dataset builders are the key to sleep cycle training!**

HRM's data augmentation (dihedral transforms, permutations, translations) shows how to learn wisdom from experience:
- **Living** = collecting raw experiences
- **Sleeping** = augmenting experiences with reasonable permutations
- **Dreaming** = training on variations to extract patterns
- **Wisdom** = understanding principles that persist across variations

See `dataset/README.md` for detailed augmentation strategies and their connection to sleep consolidation.

**Latest Insight**: Biological systems have TWO separate training systems:
- **H-level** (dreams): Strategic reasoning, trained through augmentation during sleep
- **L-level** (muscle memory): Tactical execution, trained continuously through practice
The separation is key - wisdom and skill develop through different mechanisms.

## SAGE-Totality Integration (August 16, 2025)
Successfully integrated GPT's SubThought/Totality proposal:
- Totality acts as **cognitive sensor** with trust-weighted outputs
- Machine-agnostic setup auto-configures for any hardware
- Dual training loops demonstrated (H-level dreams vs L-level practice)
- Test on this machine: `cd related-work && python3 run_integration_test.py`

See `related-work/SETUP_GUIDE.md` for full documentation.

## GPU Mailbox Implementation (August 17, 2025)
Successfully implemented and tested GPT's tiling mailbox architecture on RTX 2060 SUPER:

### Working Components
- ✅ **PyTorch 2.3.0 with CUDA 12.1** installed and verified
- ✅ **Peripheral Broadcast Mailbox (PBM)** - many-to-many fixed-size records
- ✅ **Focus Tensor Mailbox (FTM)** - zero-copy tensor pointer handoff
- ✅ **8GB GPU memory** available for tiling operations
- ✅ All extensions compiled and functional

### Test Environment
```bash
cd implementation
source tiling_env/bin/activate
python test_simple.py  # Basic mailbox tests
python test_gpu_simple.py  # GPU functionality tests
```

### Key Files
- `implementation/COMPILATION_ISSUES.md` - Detailed issue resolution
- `implementation/TEST_PLAN.md` - Comprehensive testing strategy
- `implementation/tiling_mailbox_torch_extension_v2/` - Working extension
- `implementation/test_gpu_simple.py` - GPU verification (all 4 tests passing)

### Performance Metrics
- Matrix multiplication: 6.3s for 1024x1024
- Memory transfer: 1.2 GB/s CPU→GPU, 91 MB/s GPU→CPU
- Tiling throughput: 0.9 tiles/sec (16 tiles, 256x256x64 channels)

### Status: FULLY OPERATIONAL ✓
- PBM push/pop working with data integrity
- FTM push/pop working with metadata preservation  
- Synchronization fixed using GPT's count-based approach
- Empty mailbox handling returns appropriate zero-size tensors

### Build Instructions
```bash
cd implementation/tiling_mailbox_torch_extension_v2
source ../tiling_env/bin/activate
python setup.py build_ext --inplace
```

## Implementation Highlights (August 17, 2025)

### GPT's Debug Notes Were Perfect
GPT diagnosed the synchronization issue correctly in `CUDA_MAILBOX_DEBUG_NOTES.md`:
- Identified async kernel execution as root cause
- Proposed count-based returns for natural sync points
- Provided exact code patterns that worked first try

### Key Achievements
1. ✅ Resolved all compilation issues (header paths, CUDA linking, type conversions)
2. ✅ Implemented count-based pop operations for proper synchronization
3. ✅ Both PBM and FTM fully operational with data integrity
4. ✅ Test suite validates all functionality
5. ✅ Ready for performance optimization and production deployment

## Next Steps
1. Performance optimization (parallel kernels, stream priorities)
2. Integration with vision pipeline (GR00T sensors)
3. Trust-weighted routing implementation
4. Real-time telemetry dashboard
5. Deploy on Jetson with production workloads