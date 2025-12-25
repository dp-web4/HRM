#!/usr/bin/env python3
"""
Explore available ModelOpt FP4 configurations.

This helps us understand what configs are available and their properties.
"""

import sys
from pathlib import Path

try:
    from modelopt.torch.quantization import (
        NVFP4_DEFAULT_CFG,
        NVFP4_AWQ_LITE_CFG,
        NVFP4_AWQ_FULL_CFG,
        NVFP4_MLP_WEIGHT_ONLY_CFG,
        NVFP4_SVDQUANT_DEFAULT_CFG,
    )
    import pprint

    print("="*60)
    print("AVAILABLE NVFP4 CONFIGURATIONS")
    print("="*60)

    configs = {
        "NVFP4_DEFAULT_CFG": NVFP4_DEFAULT_CFG,
        "NVFP4_AWQ_LITE_CFG": NVFP4_AWQ_LITE_CFG,
        "NVFP4_AWQ_FULL_CFG": NVFP4_AWQ_FULL_CFG,
        "NVFP4_MLP_WEIGHT_ONLY_CFG": NVFP4_MLP_WEIGHT_ONLY_CFG,
        "NVFP4_SVDQUANT_DEFAULT_CFG": NVFP4_SVDQUANT_DEFAULT_CFG,
    }

    for name, config in configs.items():
        print(f"\n{name}:")
        print("-" * 60)
        if hasattr(config, '__dict__'):
            pprint.pprint(config.__dict__, indent=2)
        else:
            pprint.pprint(config, indent=2)

    print("\n" + "="*60)
    print("WEIGHT-ONLY CONFIG DETAILS")
    print("="*60)
    print("\nNVFP4_MLP_WEIGHT_ONLY_CFG properties:")
    print(f"Type: {type(NVFP4_MLP_WEIGHT_ONLY_CFG)}")
    print(f"Content: {NVFP4_MLP_WEIGHT_ONLY_CFG}")

except ImportError as e:
    print(f"Error importing ModelOpt configs: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error exploring configs: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
