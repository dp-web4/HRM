#!/usr/bin/env python3
"""
Investigate the signatures of ModelOpt export functions.

Key functions found:
- fold_weight (from quantization module)
- export_hf_checkpoint
- export_hf_vllm_fq_checkpoint
- export_tensorrt_llm_checkpoint
"""

import inspect
from modelopt.torch.quantization import fold_weight
from modelopt.torch.export import (
    export_hf_checkpoint,
    export_hf_vllm_fq_checkpoint,
    export_tensorrt_llm_checkpoint,
)

print("="*70)
print("MODELOPT EXPORT FUNCTION SIGNATURES")
print("="*70)

# 1. fold_weight - Most promising for our case
print("\n🔧 fold_weight (from quantization):")
print("-" * 70)
sig = inspect.signature(fold_weight)
print(f"Signature: {sig}")
doc = inspect.getdoc(fold_weight)
if doc:
    print(f"\nDocumentation:\n{doc}")

# 2. export_hf_checkpoint
print("\n" + "="*70)
print("📦 export_hf_checkpoint:")
print("-" * 70)
sig = inspect.signature(export_hf_checkpoint)
print(f"Signature: {sig}")
doc = inspect.getdoc(export_hf_checkpoint)
if doc:
    print(f"\nDocumentation:\n{doc[:500]}...")

# 3. export_hf_vllm_fq_checkpoint
print("\n" + "="*70)
print("🚀 export_hf_vllm_fq_checkpoint (for vLLM with quantization):")
print("-" * 70)
sig = inspect.signature(export_hf_vllm_fq_checkpoint)
print(f"Signature: {sig}")
doc = inspect.getdoc(export_hf_vllm_fq_checkpoint)
if doc:
    print(f"\nDocumentation:\n{doc[:500]}...")

# 4. export_tensorrt_llm_checkpoint
print("\n" + "="*70)
print("⚡ export_tensorrt_llm_checkpoint:")
print("-" * 70)
sig = inspect.signature(export_tensorrt_llm_checkpoint)
print(f"Signature: {sig}")
doc = inspect.getdoc(export_tensorrt_llm_checkpoint)
if doc:
    print(f"\nDocumentation:\n{doc[:500]}...")

print("\n" + "="*70)
print("INVESTIGATION COMPLETE")
print("="*70)
print("\n💡 Key findings:")
print("1. fold_weight() - Folds quantization parameters into weights")
print("2. export_hf_checkpoint() - Standard HuggingFace export")
print("3. export_hf_vllm_fq_checkpoint() - For vLLM with FP4 quantization")
print("4. export_tensorrt_llm_checkpoint() - For TensorRT-LLM deployment")
