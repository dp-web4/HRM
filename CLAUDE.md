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

## Next Steps
1. Push changes to GitHub (waiting for PAT update)
2. Run `./install_jetson.sh` to install dependencies
3. Try the Sudoku demo with `./jetson_quick_start.sh`