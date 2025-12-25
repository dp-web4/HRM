#!/usr/bin/env python3
"""
Comprehensive diagnostic script for torch.cat() failure in FP4 quantized Qwen3-Omni-30B.

This script:
1. Loads the quantized model and inspects its structure
2. Hooks into torch.cat() to capture where it's called with empty tensors
3. Traces through the forward pass to identify the exact failure point
4. Tests minimal inference to isolate the issue
5. Compares with original model to identify quantization-specific issues
6. Provides detailed analysis and concrete solutions

Author: Claude (Diagnostic Agent)
Date: 2025-12-24
"""

import torch
import torch.nn as nn
import sys
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from contextlib import contextmanager

# Import transformers components
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor


class TorchCatMonitor:
    """Monitor all torch.cat() calls to detect empty tensor lists."""

    def __init__(self):
        self.cat_calls = []
        self.original_cat = torch.cat
        self.monitoring = False

    def start_monitoring(self):
        """Hook into torch.cat() to monitor calls."""
        self.monitoring = True
        self.cat_calls = []

        def monitored_cat(tensors, *args, **kwargs):
            """Wrapper around torch.cat that logs calls."""
            call_info = {
                'stack_trace': traceback.extract_stack(),
                'num_tensors': len(tensors) if isinstance(tensors, (list, tuple)) else 1,
                'tensor_shapes': [t.shape for t in tensors] if isinstance(tensors, (list, tuple)) and len(tensors) > 0 else [],
                'is_empty': len(tensors) == 0 if isinstance(tensors, (list, tuple)) else False,
            }

            self.cat_calls.append(call_info)

            # If empty, capture more context
            if call_info['is_empty']:
                call_info['full_traceback'] = traceback.format_stack()
                print("\n" + "="*80)
                print("⚠️  EMPTY TENSOR LIST DETECTED IN torch.cat()")
                print("="*80)
                print("Call stack (last 10 frames):")
                for frame in call_info['stack_trace'][-10:]:
                    print(f"  {frame.filename}:{frame.lineno} in {frame.name}")
                    print(f"    {frame.line}")
                print("="*80)

                # This will raise the error
                raise ValueError(
                    f"torch.cat() called with empty tensor list!\n"
                    f"Location: {call_info['stack_trace'][-2].filename}:{call_info['stack_trace'][-2].lineno}"
                )

            return self.original_cat(tensors, *args, **kwargs)

        torch.cat = monitored_cat

    def stop_monitoring(self):
        """Restore original torch.cat()."""
        torch.cat = self.original_cat
        self.monitoring = False

    def get_report(self) -> Dict[str, Any]:
        """Generate report of all torch.cat() calls."""
        return {
            'total_calls': len(self.cat_calls),
            'empty_calls': sum(1 for c in self.cat_calls if c['is_empty']),
            'calls': self.cat_calls,
        }


@contextmanager
def monitor_torch_cat():
    """Context manager for monitoring torch.cat() calls."""
    monitor = TorchCatMonitor()
    monitor.start_monitoring()
    try:
        yield monitor
    finally:
        monitor.stop_monitoring()


class ModelInspector:
    """Inspect model structure to understand quantization effects."""

    def __init__(self, model: nn.Module):
        self.model = model

    def analyze_structure(self) -> Dict[str, Any]:
        """Analyze model structure and quantization layers."""
        analysis = {
            'total_modules': 0,
            'quantized_modules': [],
            'module_types': {},
            'parameter_stats': {},
        }

        for name, module in self.model.named_modules():
            analysis['total_modules'] += 1

            # Count module types
            module_type = type(module).__name__
            analysis['module_types'][module_type] = analysis['module_types'].get(module_type, 0) + 1

            # Check for quantization layers
            if 'quantiz' in module_type.lower() or 'fp4' in module_type.lower():
                analysis['quantized_modules'].append({
                    'name': name,
                    'type': module_type,
                })

        # Analyze parameters
        total_params = 0
        quantized_params = 0

        for name, param in self.model.named_parameters():
            total_params += param.numel()

            # Check parameter dtype and properties
            param_info = {
                'name': name,
                'shape': list(param.shape),
                'dtype': str(param.dtype),
                'requires_grad': param.requires_grad,
                'numel': param.numel(),
            }

            # Check if likely quantized (FP4 would have special properties)
            if 'mlp' in name.lower() and 'weight' in name:
                quantized_params += param.numel()
                param_info['likely_quantized'] = True

            if len(analysis['parameter_stats']) < 20:  # Sample first 20 params
                analysis['parameter_stats'][name] = param_info

        analysis['total_parameters'] = total_params
        analysis['quantized_parameters'] = quantized_params
        analysis['quantization_percentage'] = (quantized_params / total_params * 100) if total_params > 0 else 0

        return analysis

    def find_attention_modules(self) -> List[str]:
        """Find all attention modules in the model."""
        attention_modules = []

        for name, module in self.model.named_modules():
            module_type = type(module).__name__
            if 'attention' in module_type.lower():
                attention_modules.append({
                    'name': name,
                    'type': module_type,
                })

        return attention_modules

    def check_forward_signature(self) -> Dict[str, Any]:
        """Check the forward method signature."""
        import inspect

        forward_method = self.model.forward
        signature = inspect.signature(forward_method)

        return {
            'parameters': [p.name for p in signature.parameters.values()],
            'signature': str(signature),
        }


class InferenceTracer:
    """Trace through inference to identify where the failure occurs."""

    def __init__(self, model: nn.Module, processor):
        self.model = model
        self.processor = processor
        self.device = next(model.parameters()).device

    def test_minimal_inference(self, prompt: str = "Hello!") -> Dict[str, Any]:
        """Test minimal inference with detailed error capture."""
        result = {
            'success': False,
            'error': None,
            'error_type': None,
            'error_location': None,
            'traceback': None,
        }

        try:
            # Prepare input
            inputs = self.processor(
                text=[prompt],
                return_tensors="pt",
            ).to(self.device)

            result['input_shape'] = {k: v.shape for k, v in inputs.items()}
            result['input_keys'] = list(inputs.keys())

            # Try single token generation
            print("Attempting single token generation...")

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                )

            result['success'] = True
            result['output_shape'] = outputs.shape
            result['output'] = self.processor.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            result['error'] = str(e)
            result['error_type'] = type(e).__name__
            result['traceback'] = traceback.format_exc()

            # Extract error location from traceback
            tb_lines = result['traceback'].split('\n')
            for line in tb_lines:
                if 'torch.cat' in line or 'File ' in line:
                    result['error_location'] = line.strip()

        return result

    def test_forward_only(self) -> Dict[str, Any]:
        """Test just the forward pass without generation."""
        result = {
            'success': False,
            'error': None,
        }

        try:
            inputs = self.processor(
                text=["Test"],
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                output = self.model(**inputs)

            result['success'] = True
            result['output_keys'] = list(output.keys()) if hasattr(output, 'keys') else []

        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()

        return result


def compare_with_original(
    quantized_model_path: str,
    original_model_path: str,
) -> Dict[str, Any]:
    """Compare quantized model structure with original."""

    print("\n" + "="*80)
    print("COMPARING QUANTIZED VS ORIGINAL MODEL STRUCTURE")
    print("="*80)

    comparison = {
        'quantized': {},
        'original': {},
        'differences': [],
    }

    # Load original model config only (to avoid memory issues)
    print("\nLoading original model config...")
    from transformers import AutoConfig

    try:
        orig_config = AutoConfig.from_pretrained(
            original_model_path,
            trust_remote_code=True,
        )

        comparison['original']['config'] = orig_config.to_dict()

    except Exception as e:
        print(f"Warning: Could not load original config: {e}")

    # Load quantized model config
    print("Loading quantized model config...")
    try:
        quant_config = AutoConfig.from_pretrained(
            quantized_model_path,
            trust_remote_code=True,
        )

        comparison['quantized']['config'] = quant_config.to_dict()

    except Exception as e:
        print(f"Warning: Could not load quantized config: {e}")

    # Compare configs
    if 'config' in comparison['original'] and 'config' in comparison['quantized']:
        for key in comparison['original']['config']:
            orig_val = comparison['original']['config'].get(key)
            quant_val = comparison['quantized']['config'].get(key)

            if orig_val != quant_val:
                comparison['differences'].append({
                    'key': key,
                    'original': orig_val,
                    'quantized': quant_val,
                })

    return comparison


def main():
    """Main diagnostic routine."""

    print("="*80)
    print("QWEN3-OMNI FP4 QUANTIZATION DIAGNOSTIC TOOL")
    print("="*80)
    print("\nThis script will:")
    print("  1. Load the quantized model")
    print("  2. Inspect model structure for quantization effects")
    print("  3. Monitor torch.cat() calls during inference")
    print("  4. Identify the exact failure point")
    print("  5. Compare with original model")
    print("  6. Provide root cause analysis and solutions")
    print("="*80)

    # Paths
    quantized_model_path = "/home/dp/ai-workspace/HRM/model-zoo/sage/omni-modal/qwen3-omni-30b-fp4-weight-only"
    original_model_path = "/home/dp/ai-workspace/HRM/model-zoo/sage/omni-modal/qwen3-omni-30b"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"\nDevice: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

    # Create diagnostic report
    diagnostic_report = {
        'device': device,
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'quantized_model_path': quantized_model_path,
        'original_model_path': original_model_path,
    }

    # Phase 1: Load quantized model
    print("\n" + "="*80)
    print("PHASE 1: LOADING QUANTIZED MODEL")
    print("="*80)

    try:
        print("\nLoading model (this may take 2-3 minutes)...")
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            quantized_model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        print("✅ Model loaded successfully")

        # Memory stats
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            print(f"GPU Memory Allocated: {allocated:.2f} GB")

        diagnostic_report['model_load'] = {'success': True}

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        diagnostic_report['model_load'] = {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }
        return

    # Load processor
    print("\nLoading processor...")
    try:
        processor = Qwen3OmniMoeProcessor.from_pretrained(quantized_model_path)
        print("✅ Processor loaded successfully")
        diagnostic_report['processor_load'] = {'success': True}
    except Exception as e:
        print(f"❌ Failed to load processor: {e}")
        diagnostic_report['processor_load'] = {
            'success': False,
            'error': str(e),
        }
        return

    # Phase 2: Inspect model structure
    print("\n" + "="*80)
    print("PHASE 2: INSPECTING MODEL STRUCTURE")
    print("="*80)

    inspector = ModelInspector(model)

    print("\nAnalyzing model structure...")
    structure_analysis = inspector.analyze_structure()

    print(f"\nModel Structure:")
    print(f"  Total modules: {structure_analysis['total_modules']}")
    print(f"  Quantized modules: {len(structure_analysis['quantized_modules'])}")
    print(f"  Total parameters: {structure_analysis['total_parameters']/1e9:.2f}B")
    print(f"  Quantized parameters: {structure_analysis['quantized_parameters']/1e9:.2f}B ({structure_analysis['quantization_percentage']:.1f}%)")

    print(f"\nModule types (top 10):")
    sorted_types = sorted(structure_analysis['module_types'].items(), key=lambda x: x[1], reverse=True)
    for module_type, count in sorted_types[:10]:
        print(f"  {module_type}: {count}")

    if structure_analysis['quantized_modules']:
        print(f"\nQuantized modules (first 5):")
        for qmod in structure_analysis['quantized_modules'][:5]:
            print(f"  {qmod['name']} ({qmod['type']})")

    diagnostic_report['structure_analysis'] = structure_analysis

    # Find attention modules
    print("\nFinding attention modules...")
    attention_modules = inspector.find_attention_modules()
    print(f"Found {len(attention_modules)} attention modules")

    if attention_modules:
        print("Attention module types:")
        for am in attention_modules[:5]:
            print(f"  {am['type']}")

    diagnostic_report['attention_modules'] = attention_modules

    # Check forward signature
    print("\nChecking forward() signature...")
    forward_sig = inspector.check_forward_signature()
    print(f"Forward parameters: {', '.join(forward_sig['parameters'])}")

    diagnostic_report['forward_signature'] = forward_sig

    # Phase 3: Test inference with monitoring
    print("\n" + "="*80)
    print("PHASE 3: TESTING INFERENCE WITH torch.cat() MONITORING")
    print("="*80)

    tracer = InferenceTracer(model, processor)

    print("\nEnabling torch.cat() monitoring...")

    try:
        with monitor_torch_cat() as monitor:
            print("Running minimal inference test...")
            inference_result = tracer.test_minimal_inference("Hello! How are you?")

            # Get monitoring report
            cat_report = monitor.get_report()

            print(f"\ntorch.cat() monitoring results:")
            print(f"  Total calls: {cat_report['total_calls']}")
            print(f"  Empty calls: {cat_report['empty_calls']}")

            diagnostic_report['cat_monitoring'] = cat_report

    except Exception as e:
        print(f"\n❌ Inference failed with error: {e}")
        inference_result = {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }

    diagnostic_report['inference_test'] = inference_result

    # Print inference results
    if inference_result['success']:
        print("\n✅ Inference test PASSED!")
        print(f"Output: {inference_result.get('output', 'N/A')}")
    else:
        print("\n❌ Inference test FAILED!")
        print(f"Error type: {inference_result.get('error_type', 'Unknown')}")
        print(f"Error message: {inference_result.get('error', 'N/A')}")

        if inference_result.get('error_location'):
            print(f"Error location: {inference_result['error_location']}")

        if inference_result.get('traceback'):
            print("\nFull traceback:")
            print(inference_result['traceback'])

    # Phase 4: Test forward only
    print("\n" + "="*80)
    print("PHASE 4: TESTING FORWARD PASS ONLY (NO GENERATION)")
    print("="*80)

    forward_result = tracer.test_forward_only()
    diagnostic_report['forward_test'] = forward_result

    if forward_result['success']:
        print("✅ Forward pass PASSED!")
        print(f"Output keys: {forward_result.get('output_keys', [])}")
    else:
        print("❌ Forward pass FAILED!")
        print(f"Error: {forward_result.get('error', 'N/A')}")

    # Phase 5: Compare with original
    print("\n" + "="*80)
    print("PHASE 5: COMPARING WITH ORIGINAL MODEL")
    print("="*80)

    comparison = compare_with_original(quantized_model_path, original_model_path)
    diagnostic_report['model_comparison'] = comparison

    if comparison['differences']:
        print(f"\nFound {len(comparison['differences'])} config differences:")
        for diff in comparison['differences'][:10]:
            print(f"  {diff['key']}:")
            print(f"    Original: {diff['original']}")
            print(f"    Quantized: {diff['quantized']}")

    # Phase 6: Root cause analysis
    print("\n" + "="*80)
    print("PHASE 6: ROOT CAUSE ANALYSIS")
    print("="*80)

    analysis = []

    # Analyze the failure
    if not inference_result['success']:
        error_msg = inference_result.get('error', '')

        if 'torch.cat' in error_msg and 'empty' in error_msg.lower():
            analysis.append({
                'finding': 'Empty tensor list in torch.cat()',
                'severity': 'CRITICAL',
                'description': 'torch.cat() is being called with an empty list of tensors during generation',
                'likely_cause': 'Qwen3-Omni multimodal architecture incompatibility with quantization',
            })

        if 'attention' in error_msg.lower():
            analysis.append({
                'finding': 'Attention mechanism issue',
                'severity': 'HIGH',
                'description': 'The error may be related to attention computation',
                'likely_cause': 'Quantization affecting attention KV-cache or multi-head attention concatenation',
            })

    # Check quantization metadata
    metadata_path = Path(quantized_model_path) / "quantization_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

        if metadata.get('compression_ratio', 1.0) == 1.0:
            analysis.append({
                'finding': 'No actual compression achieved',
                'severity': 'CRITICAL',
                'description': f"Compression ratio is {metadata['compression_ratio']}x (expected 2-3x)",
                'likely_cause': 'Quantization did not actually compress the weights - this is expected for runtime quantization',
            })

    diagnostic_report['root_cause_analysis'] = analysis

    print("\nFindings:")
    for i, finding in enumerate(analysis, 1):
        print(f"\n{i}. {finding['finding']} [{finding['severity']}]")
        print(f"   Description: {finding['description']}")
        print(f"   Likely cause: {finding['likely_cause']}")

    # Phase 7: Recommendations
    print("\n" + "="*80)
    print("PHASE 7: RECOMMENDATIONS")
    print("="*80)

    recommendations = []

    if not inference_result['success']:
        if 'torch.cat' in inference_result.get('error', ''):
            recommendations.append({
                'priority': 'HIGH',
                'recommendation': 'Investigate Qwen3-Omni multimodal architecture',
                'action': 'Check if the model has multiple forward paths (text, audio, vision) and if quantization breaks the routing',
                'code_example': '''
# Check model architecture for multimodal components
for name, module in model.named_modules():
    if 'audio' in name.lower() or 'vision' in name.lower():
        print(f"Multimodal component: {name}")
'''
            })

            recommendations.append({
                'priority': 'HIGH',
                'recommendation': 'Test with text-only inputs',
                'action': 'Verify if the issue is specific to multimodal inputs or affects all inputs',
                'code_example': '''
# Try pure text generation without any audio/vision inputs
inputs = processor(text=["Hello"], return_tensors="pt", audio=None, images=None)
outputs = model.generate(**inputs, max_new_tokens=5)
'''
            })

            recommendations.append({
                'priority': 'MEDIUM',
                'recommendation': 'Use vLLM for runtime quantization',
                'action': 'Since this is a runtime quantization model, use vLLM which is designed for this',
                'code_example': '''
# Load with vLLM (handles FP4 runtime quantization)
from vllm import LLM

llm = LLM(
    model=quantized_model_path,
    quantization="fp4",  # vLLM will quantize at runtime
    tensor_parallel_size=1,
)
'''
            })

    diagnostic_report['recommendations'] = recommendations

    print("\nRecommended actions:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. [{rec['priority']}] {rec['recommendation']}")
        print(f"   Action: {rec['action']}")
        if rec.get('code_example'):
            print(f"   Example code:")
            for line in rec['code_example'].strip().split('\n'):
                print(f"     {line}")

    # Save diagnostic report
    report_path = Path("/home/dp/ai-workspace/HRM/sage/quantization/diagnostic_report.json")
    print(f"\n\nSaving diagnostic report to: {report_path}")

    with open(report_path, 'w') as f:
        json.dump(diagnostic_report, f, indent=2, default=str)

    print("✅ Diagnostic report saved")

    # Summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)

    print(f"\n✅ Model loaded: {diagnostic_report['model_load']['success']}")
    print(f"✅ Processor loaded: {diagnostic_report['processor_load']['success']}")
    print(f"{'✅' if inference_result['success'] else '❌'} Inference test: {'PASSED' if inference_result['success'] else 'FAILED'}")
    print(f"{'✅' if forward_result['success'] else '❌'} Forward test: {'PASSED' if forward_result['success'] else 'FAILED'}")

    print(f"\nTotal findings: {len(analysis)}")
    print(f"Total recommendations: {len(recommendations)}")

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("\n1. Review the diagnostic report: sage/quantization/diagnostic_report.json")
    print("2. Examine the error traceback to find exact torch.cat() location")
    print("3. Consider using vLLM for runtime quantization instead of direct HF loading")
    print("4. Test if the issue is multimodal-specific or affects all inputs")
    print("5. Check ModelOpt documentation for Qwen3-Omni-specific quantization notes")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
