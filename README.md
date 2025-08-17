# HRM - SAGE Integration Experiments

> **Experimental Fork**: This is an experimental fork of the original [Hierarchical Reasoning Model (HRM)](https://github.com/SapientAI-Inc/hierarchical-reasoning-model) by Sapient AI, used as a foundation for SAGE (Sentient Agentic Generative Engine) research.

## Project Status

⚠️ **EXPERIMENTAL - DISCOVERY MODE** ⚠️

This repository is in active discovery mode, exploring the integration of HRM into a broader consciousness architecture. We are:
- Testing philosophical hypotheses about emergent coherence
- Building GPU-resident infrastructure for zero-copy module communication
- Integrating multiple cognitive subsystems (HRM, Sidecar Memory, Totality/SubThought, GR00T)
- Measuring state evolution and coherence emergence

**Not for production use.** Stubs, placeholders, and experimental code are expected and marked accordingly.

## Conceptual Framework

This work operates within a three-layer conceptual stack:

```
Philosophy → Governance → Implementation
Synchronism → Web4 → SAGE
```

- **[Synchronism](https://dpcars.net/synchronism)**: Coherence emerges from resonance across scales of intent
- **[Web4](https://metalinxx.io/web4_whitepaper/)**: Trust-weighted governance of information flow
- **[SAGE](./SAGE_WHITEPAPER.md)**: Learned coherence replacing programmed rules

See [CROSSREF_APPENDIX.md](./implementation/CROSSREF_APPENDIX.md) for detailed framework integration.

## Original HRM

![](./assets/hrm.png)

The original Hierarchical Reasoning Model by Sapient AI is a groundbreaking 27M parameter model that:
- Achieves near-perfect performance on complex reasoning tasks with only 1000 training samples
- Operates without pre-training or Chain-of-Thought data
- Solves extreme Sudoku puzzles and large maze pathfinding
- Outperforms much larger models on the Abstraction and Reasoning Corpus (ARC)

Original paper: [arXiv:2506.21734](https://arxiv.org/abs/2506.21734)

## Our Extensions

### 1. GPU Mailbox Architecture
Zero-copy communication substrate for module intercommunication. See [GPU_MAILBOX.md](./implementation/GPU_MAILBOX.md).

### 2. SAGE Integration
- Trust-weighted sensor fusion across physical, temporal, and cognitive sensors
- Dual memory systems: H-level (strategic/dreams) and L-level (tactical/practice)
- Sleep consolidation and augmentation-based learning

### 3. GR00T Integration
Embodied intelligence through NVIDIA's GR00T foundation model. See [gr00t-integration/](./gr00t-integration/).

### 4. Totality/SubThought Integration
Cognitive sensor providing structured imagination. See [related-work/](./related-work/).

## Implementation Philosophy

We follow a **Discovery vs Delivery** paradigm:
- **Discovery Mode** (this repo): Prove concepts, measure effects, validate hypotheses
- **Delivery Mode** (not this repo): Production-ready, fully tested, optimized

Key principles:
- ✅ **Implemented (minimal)**: Works but intentionally simple
- 🧪 **Stub/Placeholder**: Enables testing but must be replaced
- ⚠️ **Known wrinkle**: Edge cases or gaps to address

See [STUB_ALERT_APPENDIX.md](./related-work/STUB_ALERT_APPENDIX.md) for current stub status.

## Documentation

- **[IMPLEMENTATION_LOG.md](./implementation/IMPLEMENTATION_LOG.md)**: Append-only log of implementation decisions and discoveries
- **[IMPLEMENTATION_README.md](./implementation/IMPLEMENTATION_README.md)**: System philosophy and integration approach
- **[ENGINEERING_PRINCIPLES.md](../private-context/ENGINEERING_PRINCIPLES.md)**: Discovery vs Delivery paradigm details

---

## Original HRM Documentation

Below is the original documentation from Sapient AI's HRM repository:

## Quick Start Guide 🚀

### Prerequisites ⚙️

Ensure PyTorch and CUDA are installed. The repo needs CUDA extensions to be built. If not present, run the following commands:

```bash
# Install CUDA 12.6
CUDA_URL=https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run

wget -q --show-progress --progress=bar:force:noscroll -O cuda_installer.run $CUDA_URL
sudo sh cuda_installer.run --silent --toolkit --override

export CUDA_HOME=/usr/local/cuda-12.6

# Install PyTorch with CUDA 12.6
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

pip3 install torch torchvision torchaudio --index-url $PYTORCH_INDEX_URL

# Additional packages for building extensions
pip3 install packaging ninja wheel setuptools setuptools-scm
```

Then install FlashAttention. For Hopper GPUs, install FlashAttention 3

```bash
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```

For Ampere or earlier GPUs, install FlashAttention 2

```bash
pip3 install flash-attn
```

#### Jetson Orin Nano Setup

The Jetson Orin Nano requires special consideration due to its ARM architecture and memory constraints:

**Option 1: Docker (Recommended for Quick Start)**
```bash
# Use NVIDIA's pre-built PyTorch container
sudo docker run --rm --runtime nvidia \
    -v $(pwd):/workspace \
    dustynv/l4t-pytorch:r36.2.0 \
    python3 /workspace/your_script.py
```

**Option 2: Build from Source (For Native Performance)**

Due to cuDNN version mismatches and memory limitations, building PyTorch and Flash Attention from source is recommended:

1. **Create swap file on SSD (REQUIRED):**
```bash
# Minimum 16GB swap, 24GB recommended
sudo fallocate -l 24G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent (add to /etc/fstab):
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

2. **Build PyTorch from source:**
```bash
# Clone PyTorch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout v2.5.0

# Configure for Jetson (SM 8.7)
export USE_CUDA=1
export USE_CUDNN=1
export TORCH_CUDA_ARCH_LIST="8.7"
export CUDA_HOME=/usr/local/cuda-12.6
export MAX_JOBS=2  # Limit parallel jobs to avoid OOM

# Build (takes 4-6 hours)
python3 setup.py develop --user
```

3. **Build Flash Attention:**
```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.3.6  # More stable for Jetson

export MAX_JOBS=1  # Very conservative for Flash Attention
python3 setup.py build_ext --inplace
python3 setup.py install --user
```

**Note:** Flash Attention compilation is extremely memory-intensive. The 24GB swap is essential to avoid OOM kills during the CUDA kernel compilation.

## Install Python Dependencies 🐍

```bash
pip install -r requirements.txt
```

## W&B Integration 📈

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking and metric visualization. Ensure you're logged in:

```bash
wandb login
```

## Run Experiments

### Quick Demo: Sudoku Solver 💻🗲

Train a master-level Sudoku AI capable of solving extremely difficult puzzles on a modern laptop GPU. 🧩

```bash
# Download and build Sudoku dataset
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000

# Start training (single GPU, smaller batch size)
OMP_NUM_THREADS=8 python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 global_batch_size=384 lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

Runtime: ~10 hours on a RTX 4070 laptop GPU

## Trained Checkpoints 🚧

 - [ARC-AGI-2](https://huggingface.co/sapientinc/HRM-checkpoint-ARC-2)
 - [Sudoku 9x9 Extreme (1000 examples)](https://huggingface.co/sapientinc/HRM-checkpoint-sudoku-extreme)
 - [Maze 30x30 Hard (1000 examples)](https://huggingface.co/sapientinc/HRM-checkpoint-maze-30x30-hard)

To use the checkpoints, see Evaluation section below.

## Full-scale Experiments 🔵

Experiments below assume an 8-GPU setup.

### Dataset Preparation

```bash
# Initialize submodules
git submodule update --init --recursive

# ARC-1
python dataset/build_arc_dataset.py  # ARC offical + ConceptARC, 960 examples
# ARC-2
python dataset/build_arc_dataset.py --dataset-dirs dataset/raw-data/ARC-AGI-2/data --output-dir data/arc-2-aug-1000  # ARC-2 official, 1120 examples

# Sudoku-Extreme
python dataset/build_sudoku_dataset.py  # Full version
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  # 1000 examples

# Maze
python dataset/build_maze_dataset.py  # 1000 examples
```

### Dataset Visualization

Explore the puzzles visually:

* Open `puzzle_visualizer.html` in your browser.
* Upload the generated dataset folder located in `data/...`.

## Launch experiments

### Small-sample (1K)

ARC-1:

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py 
```

*Runtime:* ~24 hours

ARC-2:

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/arc-2-aug-1000
```

*Runtime:* ~24 hours (checkpoint after 8 hours is often sufficient)

Sudoku Extreme (1k):

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

*Runtime:* ~10 minutes

Maze 30x30 Hard (1k):

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/maze-30x30-hard-1k epochs=20000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

*Runtime:* ~1 hour

### Full Sudoku-Hard

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/sudoku-hard-full epochs=100 eval_interval=10 lr_min_ratio=0.1 global_batch_size=2304 lr=3e-4 puzzle_emb_lr=3e-4 weight_decay=0.1 puzzle_emb_weight_decay=0.1 arch.loss.loss_type=softmax_cross_entropy arch.L_cycles=8 arch.halt_max_steps=8 arch.pos_encodings=learned
```

*Runtime:* ~2 hours

## Evaluation

Evaluate your trained models:

* Check `eval/exact_accuracy` in W&B.
* For ARC-AGI, follow these additional steps:

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 evaluate.py checkpoint=<CHECKPOINT_PATH>
```

* Then use the provided `arc_eval.ipynb` notebook to finalize and inspect your results.

## Notes

 - Small-sample learning typically exhibits accuracy variance of around ±2 points.
 - For Sudoku-Extreme (1,000-example dataset), late-stage overfitting may cause numerical instability during training and Q-learning. It is advisable to use early stopping once the training accuracy approaches 100%.

## Citation 📜

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```