# SAGE Raising Edge Notes

## Edge Model Path Configuration

The IntrospectiveQwenIRP uses a hardcoded path to Thor's model directory.
For Sprout (edge) operation, override the model path.

### Thor Path (default)
```
/home/dp/ai-workspace/HRM/sage/experiments/phase1-hierarchical-cognitive/epistemic_bias_mapping/Introspective-Qwen-0.5B-v2.1/model
```

### Sprout Path
```
/home/sprout/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/Introspective-Qwen-0.5B-v2.1/model
```

### Configuration
Pass the model path via config when creating the IRP:
```python
config = {'model_path': '/home/sprout/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/Introspective-Qwen-0.5B-v2.1/model'}
irp = IntrospectiveQwenIRP(config=config)
```

## Edge Performance Validated (2026-01-14)

- Model loads successfully on Jetson Orin Nano 8GB
- Inference works with Qwen 0.5B + LoRA adapter
- Memory usage acceptable for edge operation
- Response generation operational

## TODO for Edge Raising

1. Add environment variable or config file for model path
2. Test full programmatic session on edge
3. Consider edge-specific session cadence (power/thermal constraints)
