#!/usr/bin/env python3
"""
Investigate ModelOpt export/save methods for FP4 quantization.

We successfully applied quantization to 92.4% of parameters, but the
save_pretrained() method didn't compress the weights on disk.

Let's find the correct way to export FP4 models.
"""

import inspect
import modelopt.torch.quantization as mtq

print("="*70)
print("MODELOPT QUANTIZATION MODULE INSPECTION")
print("="*70)

# List all functions/classes in the quantization module
print("\n📦 Available functions and classes:")
print("-" * 70)
members = inspect.getmembers(mtq)
for name, obj in members:
    if not name.startswith('_'):
        if inspect.isfunction(obj) or inspect.isclass(obj):
            print(f"  {name}: {type(obj).__name__}")

# Look for export/save related functions
print("\n💾 Export/Save related functions:")
print("-" * 70)
export_related = [name for name, obj in members
                  if any(keyword in name.lower()
                        for keyword in ['export', 'save', 'convert', 'deploy', 'onnx', 'checkpoint'])]
for name in export_related:
    print(f"  ✓ {name}")

# Check if there's a model_opt or export submodule
print("\n📁 Submodules:")
print("-" * 70)
try:
    import modelopt.torch.export as export_module
    print("  ✓ modelopt.torch.export found!")
    export_members = inspect.getmembers(export_module)
    for name, obj in export_members:
        if not name.startswith('_') and (inspect.isfunction(obj) or inspect.isclass(obj)):
            print(f"    - {name}")
except ImportError:
    print("  ✗ No modelopt.torch.export module")

try:
    import modelopt.torch.opt as opt_module
    print("  ✓ modelopt.torch.opt found!")
    opt_members = inspect.getmembers(opt_module)
    for name, obj in opt_members:
        if not name.startswith('_') and (inspect.isfunction(obj) or inspect.isclass(obj)):
            print(f"    - {name}")
except ImportError:
    print("  ✗ No modelopt.torch.opt module")

# Check the quantize function signature
print("\n🔍 'quantize' function signature:")
print("-" * 70)
sig = inspect.signature(mtq.quantize)
print(f"  {sig}")

# Check for any conversion utilities
print("\n🔧 Looking for conversion utilities:")
print("-" * 70)
conversion_funcs = [name for name, obj in members
                    if 'convert' in name.lower() or 'materialize' in name.lower()]
for name in conversion_funcs:
    obj = getattr(mtq, name)
    if callable(obj):
        try:
            sig = inspect.signature(obj)
            print(f"  {name}{sig}")
            doc = inspect.getdoc(obj)
            if doc:
                print(f"    {doc[:100]}...")
        except:
            print(f"  {name}")

# Look for model export/deployment functions
print("\n🚀 Deployment/Export functions:")
print("-" * 70)
try:
    # Try to find functions related to deployment
    import modelopt
    all_modules = dir(modelopt.torch)
    print(f"  Available torch modules: {all_modules}")
except Exception as e:
    print(f"  Error exploring modules: {e}")

# Check documentation for quantize function
print("\n📚 'quantize' function documentation:")
print("-" * 70)
doc = inspect.getdoc(mtq.quantize)
if doc:
    print(doc[:500])
else:
    print("  No documentation found")

print("\n" + "="*70)
print("INVESTIGATION COMPLETE")
print("="*70)
