#!/usr/bin/env python3
"""
Qwen3-Omni Expert Extraction Tool

Extracts individual MoE experts from monolithic safetensors shards
for SAGE's selective expert loading.

Based on analysis of Qwen3-Omni-30B architecture:
- Thinker: 48 layers × 128 experts (shards 1-13)
- Talker: 20 layers × 128 experts (shard 14)
- Each expert: 3 weights (gate_proj, up_proj, down_proj) = 9MB

Usage:
    python expert_extractor.py --help
    python expert_extractor.py --extract-all
    python expert_extractor.py --extract-expert 0 --layer 0
    python expert_extractor.py --extract-routers
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import safetensors
import safetensors.torch
import torch
from tqdm import tqdm


class ExpertExtractor:
    """Extract and manage individual experts from Qwen3-Omni safetensors"""

    # Shard boundaries discovered via analysis
    THINKER_SHARDS = range(1, 14)  # Shards 1-13
    TALKER_SHARDS = [14]  # Shard 14
    OTHER_SHARDS = [15]  # Shard 15 (vision, audio, etc.)

    THINKER_LAYERS = 48
    TALKER_LAYERS = 20
    EXPERTS_PER_LAYER = 128

    def __init__(self, model_path: str, output_dir: str):
        """
        Args:
            model_path: Path to model-zoo/sage/omni-modal/qwen3-omni-30b/
            output_dir: Where to save extracted experts
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.experts_dir = self.output_dir / "experts"
        self.routers_dir = self.output_dir / "routers"
        self.experts_dir.mkdir(exist_ok=True)
        self.routers_dir.mkdir(exist_ok=True)

        # Metadata
        self.manifest = {
            "model": "Qwen3-Omni-30B-A3B-Instruct",
            "extraction_date": None,
            "thinker": {
                "layers": self.THINKER_LAYERS,
                "experts_per_layer": self.EXPERTS_PER_LAYER,
                "expert_size_mb": 9.0,
            },
            "talker": {
                "layers": self.TALKER_LAYERS,
                "experts_per_layer": self.EXPERTS_PER_LAYER,
                "expert_size_mb": 4.0,
            },
            "experts": [],
            "routers": []
        }

    def get_shard_path(self, shard_num: int) -> Path:
        """Get path to specific shard"""
        return self.model_path / f"model-{shard_num:05d}-of-00015.safetensors"

    def extract_expert(
        self,
        expert_id: int,
        layer_id: int,
        component: str = "thinker",
        force: bool = False
    ) -> Optional[Path]:
        """
        Extract a single expert to file

        Args:
            expert_id: Expert ID (0-127)
            layer_id: Layer ID (0-47 for thinker, 0-19 for talker)
            component: "thinker" or "talker"
            force: Overwrite if already exists

        Returns:
            Path to extracted expert file, or None if failed
        """
        # Determine output filename
        output_file = self.experts_dir / f"{component}_expert_{expert_id:03d}_layer_{layer_id:02d}.safetensors"

        if output_file.exists() and not force:
            print(f"Expert already exists: {output_file.name}")
            return output_file

        # Find which shard contains this layer
        shard_num = self._find_shard_for_layer(layer_id, component)
        if shard_num is None:
            print(f"❌ Layer {layer_id} not found for {component}")
            return None

        shard_path = self.get_shard_path(shard_num)

        # Extract expert weights
        expert_weights = {}

        try:
            with safetensors.safe_open(shard_path, framework="pt") as f:
                # Key pattern: {component}.model.layers.{layer}.mlp.experts.{expert_id}.{proj}.weight
                prefix = f"{component}.model.layers.{layer_id}.mlp.experts.{expert_id}."

                for key in f.keys():
                    if prefix in key:
                        expert_weights[key] = f.get_tensor(key)

                if len(expert_weights) == 0:
                    print(f"❌ No weights found for expert {expert_id} in layer {layer_id}")
                    return None

                if len(expert_weights) != 3:
                    print(f"⚠️  Expected 3 weights, found {len(expert_weights)}")

                # Save expert
                safetensors.torch.save_file(expert_weights, output_file)

                # Calculate size
                size_mb = output_file.stat().st_size / 1024**2

                # Update manifest
                self.manifest["experts"].append({
                    "component": component,
                    "expert_id": expert_id,
                    "layer_id": layer_id,
                    "file": output_file.name,
                    "size_mb": size_mb,
                    "num_weights": len(expert_weights)
                })

                print(f"✅ Extracted {component} expert {expert_id} layer {layer_id} ({size_mb:.1f} MB)")
                return output_file

        except Exception as e:
            print(f"❌ Error extracting expert: {e}")
            return None

    def extract_all_experts(self, component: str = "thinker", force: bool = False):
        """
        Extract all experts for a component

        Args:
            component: "thinker" or "talker"
            force: Overwrite existing files
        """
        num_layers = self.THINKER_LAYERS if component == "thinker" else self.TALKER_LAYERS
        total = num_layers * self.EXPERTS_PER_LAYER

        print(f"\n{'='*60}")
        print(f"Extracting {total} {component} experts ({num_layers} layers × {self.EXPERTS_PER_LAYER} experts)")
        print(f"{'='*60}\n")

        with tqdm(total=total, desc=f"{component.capitalize()} experts") as pbar:
            for layer_id in range(num_layers):
                for expert_id in range(self.EXPERTS_PER_LAYER):
                    self.extract_expert(expert_id, layer_id, component, force)
                    pbar.update(1)

        print(f"\n✅ Completed {component} expert extraction\n")

    def extract_router(self, layer_id: int, component: str = "thinker", force: bool = False) -> Optional[Path]:
        """
        Extract router weights for a layer

        Router is lightweight: [128, 2048] = 512KB per layer

        Args:
            layer_id: Layer ID
            component: "thinker" or "talker"
            force: Overwrite if exists

        Returns:
            Path to router file
        """
        output_file = self.routers_dir / f"{component}_router_layer_{layer_id:02d}.safetensors"

        if output_file.exists() and not force:
            print(f"Router already exists: {output_file.name}")
            return output_file

        # Find shard
        shard_num = self._find_shard_for_layer(layer_id, component)
        if shard_num is None:
            return None

        shard_path = self.get_shard_path(shard_num)

        try:
            with safetensors.safe_open(shard_path, framework="pt") as f:
                # Key: {component}.model.layers.{layer}.mlp.gate.weight
                router_key = f"{component}.model.layers.{layer_id}.mlp.gate.weight"

                if router_key not in f.keys():
                    print(f"❌ Router not found: {router_key}")
                    return None

                router_weights = {router_key: f.get_tensor(router_key)}

                # Save
                safetensors.torch.save_file(router_weights, output_file)

                size_mb = output_file.stat().st_size / 1024**2

                # Update manifest
                self.manifest["routers"].append({
                    "component": component,
                    "layer_id": layer_id,
                    "file": output_file.name,
                    "size_mb": size_mb
                })

                print(f"✅ Extracted {component} router layer {layer_id} ({size_mb:.1f} MB)")
                return output_file

        except Exception as e:
            print(f"❌ Error extracting router: {e}")
            return None

    def extract_all_routers(self, component: str = "thinker", force: bool = False):
        """Extract all routers for a component"""
        num_layers = self.THINKER_LAYERS if component == "thinker" else self.TALKER_LAYERS

        print(f"\n{'='*60}")
        print(f"Extracting {num_layers} {component} routers")
        print(f"{'='*60}\n")

        for layer_id in tqdm(range(num_layers), desc=f"{component.capitalize()} routers"):
            self.extract_router(layer_id, component, force)

        print(f"\n✅ Completed {component} router extraction\n")

    def _find_shard_for_layer(self, layer_id: int, component: str) -> Optional[int]:
        """
        Find which shard contains a specific layer

        Based on discovered distribution:
        - Thinker layers 0-47: Shards 1-13
        - Talker layers 0-19: Shard 14
        """
        if component == "thinker":
            if layer_id < 0 or layer_id >= self.THINKER_LAYERS:
                return None

            # Approximate shard mapping (layers distributed roughly evenly)
            # Refined mapping could be discovered via full scan
            if layer_id <= 1:
                return 1
            elif layer_id <= 5:
                return 2
            elif layer_id <= 9:
                return 3
            elif layer_id <= 13:
                return 4
            elif layer_id <= 17:
                return 5
            elif layer_id <= 21:
                return 6
            elif layer_id <= 25:
                return 7
            elif layer_id <= 29:
                return 8
            elif layer_id <= 33:
                return 9
            elif layer_id <= 37:
                return 10
            elif layer_id <= 41:
                return 11
            elif layer_id <= 45:
                return 12
            else:
                return 13

        elif component == "talker":
            if layer_id < 0 or layer_id >= self.TALKER_LAYERS:
                return None
            return 14  # All talker layers in shard 14

        return None

    def save_manifest(self):
        """Save extraction manifest with metadata"""
        import datetime
        self.manifest["extraction_date"] = datetime.datetime.now().isoformat()

        manifest_path = self.output_dir / "extraction_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

        print(f"\n✅ Manifest saved: {manifest_path}")

        # Print summary
        print(f"\n{'='*60}")
        print("Extraction Summary")
        print(f"{'='*60}")
        print(f"Total experts extracted: {len(self.manifest['experts'])}")
        print(f"Total routers extracted: {len(self.manifest['routers'])}")

        total_size = sum(e["size_mb"] for e in self.manifest["experts"])
        total_size += sum(r["size_mb"] for r in self.manifest["routers"])

        print(f"Total size: {total_size:.1f} MB")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Extract Qwen3-Omni MoE experts")

    parser.add_argument(
        "--model-path",
        default="model-zoo/sage/omni-modal/qwen3-omni-30b",
        help="Path to model directory"
    )

    parser.add_argument(
        "--output-dir",
        default="model-zoo/sage/omni-modal/qwen3-omni-30b-extracted",
        help="Output directory for extracted experts"
    )

    parser.add_argument(
        "--extract-all",
        action="store_true",
        help="Extract all experts and routers"
    )

    parser.add_argument(
        "--extract-expert",
        type=int,
        metavar="EXPERT_ID",
        help="Extract specific expert (0-127)"
    )

    parser.add_argument(
        "--layer",
        type=int,
        help="Layer ID for --extract-expert"
    )

    parser.add_argument(
        "--component",
        choices=["thinker", "talker"],
        default="thinker",
        help="Component to extract from"
    )

    parser.add_argument(
        "--extract-routers",
        action="store_true",
        help="Extract all routers"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files"
    )

    args = parser.parse_args()

    # Create extractor
    extractor = ExpertExtractor(args.model_path, args.output_dir)

    # Execute requested operation
    if args.extract_all:
        # Extract everything
        extractor.extract_all_routers("thinker", args.force)
        extractor.extract_all_routers("talker", args.force)
        extractor.extract_all_experts("thinker", args.force)
        extractor.extract_all_experts("talker", args.force)

    elif args.extract_routers:
        extractor.extract_all_routers(args.component, args.force)

    elif args.extract_expert is not None:
        if args.layer is None:
            parser.error("--layer required with --extract-expert")
        extractor.extract_expert(args.extract_expert, args.layer, args.component, args.force)

    else:
        parser.print_help()
        return

    # Save manifest
    extractor.save_manifest()


if __name__ == "__main__":
    main()
