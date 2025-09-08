#!/usr/bin/env python3
"""
Minimal test to verify SAGE V2 actually works.
Let's test the absolute minimum functionality.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

def test_minimal():
    """Test absolute minimum - can we create and run the model?"""
    print("Testing minimal SAGE V2 functionality...")
    
    try:
        # Import with minimal config
        from core.sage_v2 import SAGEV2Core, SAGEV2Config
        
        # Create tiny config
        config = SAGEV2Config(
            hidden_size=64,
            num_h_layers=1,
            num_l_layers=1,
            num_heads=4,  # 64 / 4 = 16, should work
            intermediate_size=128,
            use_external_llm=False,  # No LLM for minimal test
            use_meaningful_context=False,  # Disable complex context
            use_temporal_context=False,
            use_memory_context=False
        )
        
        print(f"Config created: hidden={config.hidden_size}, heads={config.num_heads}")
        
        # Create model
        model = SAGEV2Core(config)
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✅ Model created: {param_count:.3f}M parameters")
        
        # Test forward pass
        batch_size = 2
        input_grid = torch.randint(0, 10, (batch_size, 5, 5))
        
        with torch.no_grad():
            output = model(input_grid, num_rounds=1)
        
        print(f"✅ Forward pass successful!")
        print(f"   Output keys: {list(output.keys())}")
        
        if 'logits' in output:
            print(f"   Logits shape: {output['logits'].shape}")
            
            # Check output diversity
            predictions = output['logits'].argmax(dim=-1)
            unique_values = len(torch.unique(predictions))
            print(f"   Unique output values: {unique_values}")
            
            if unique_values == 1:
                print("⚠️  Warning: Model outputting constant values")
                return False
            else:
                print("✅ Model produces diverse outputs!")
                return True
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_minimal()
    if success:
        print("\n🎉 Minimal test passed! Core functionality works.")
    else:
        print("\n❌ Minimal test failed. Need to fix basic issues.")
    
    exit(0 if success else 1)