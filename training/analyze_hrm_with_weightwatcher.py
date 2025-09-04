#!/usr/bin/env python3
"""
Analyze HRM models using WeightWatcher
"""
import sys
sys.path.append('/home/dp/ai-workspace/weightwatcher')
import weightwatcher as ww
import torch
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Import our model
sys.path.append('.')
from evaluate_original_arc import HierarchicalReasoningModule, MODEL_CONFIG

def analyze_model(checkpoint_path, model_name):
    """Analyze a specific model checkpoint with WeightWatcher"""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    # Load model
    device = torch.device('cpu')  # WeightWatcher works on CPU
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model with appropriate config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = MODEL_CONFIG
        
    model = HierarchicalReasoningModule(config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Analyze with WeightWatcher
    watcher = ww.WeightWatcher(model=model)
    details = watcher.analyze()
    
    # Summary statistics
    summary = watcher.get_summary(details)
    
    print(f"\nWeightWatcher Analysis Summary:")
    alpha = summary.get('alpha', None)
    alpha_weighted = summary.get('alpha_weighted', None)
    log_norm = summary.get('log_norm', None)
    log_alpha_norm = summary.get('log_alpha_norm', None)
    spectral_norm = summary.get('spectral_norm', None)
    mp_softrank = summary.get('mp_softrank', None)
    
    print(f"  Alpha (Heavy-tail): {alpha:.3f}" if alpha is not None else "  Alpha (Heavy-tail): N/A")
    print(f"  Alpha weighted: {alpha_weighted:.3f}" if alpha_weighted is not None else "  Alpha weighted: N/A")
    print(f"  Log norm: {log_norm:.3f}" if log_norm is not None else "  Log norm: N/A")
    print(f"  Log alpha norm: {log_alpha_norm:.3f}" if log_alpha_norm is not None else "  Log alpha norm: N/A")
    print(f"  Spectral norm: {spectral_norm:.3f}" if spectral_norm is not None else "  Spectral norm: N/A")
    print(f"  MP softrank: {mp_softrank:.3f}" if mp_softrank is not None else "  MP softrank: N/A")
    
    # Layer-wise analysis
    print(f"\nLayer Analysis:")
    print(f"{'Layer':<30} {'Alpha':<8} {'LogNorm':<8} {'Warning':<15}")
    print("-" * 65)
    
    for _, row in details.iterrows():
        layer_id = row['layer_id']
        alpha = row.get('alpha', 0)
        log_norm = row.get('log_norm', 0)
        warning = row.get('warning', '')
        
        print(f"{layer_id:<30} {alpha:<8.3f} {log_norm:<8.3f} {warning:<15}")
    
    return details, summary

def compare_models():
    """Compare multiple model checkpoints"""
    models_to_analyze = [
        ('checkpoints/hrm_arc_best.pt', 'Original (7K steps)'),
        ('checkpoints/hrm_reasoning_agi1_step_3000.pt', 'Broken Fine-tune (3K)'),
        ('checkpoints/hrm_reasoning_agi1_fixed_final.pt', 'Fixed Fine-tune (10K)'),
    ]
    
    print("WeightWatcher Analysis of HRM Models")
    print("Comparing original vs fine-tuned models")
    print("Looking for signs of over-training or degradation")
    
    results = []
    
    for checkpoint_path, model_name in models_to_analyze:
        try:
            details, summary = analyze_model(checkpoint_path, model_name)
            if summary:
                results.append({
                    'model': model_name,
                    'alpha': summary.get('alpha', 0),
                    'alpha_weighted': summary.get('alpha_weighted', 0),
                    'log_norm': summary.get('log_norm', 0),
                    'spectral_norm': summary.get('spectral_norm', 0),
                    'mp_softrank': summary.get('mp_softrank', 0),
                })
        except Exception as e:
            print(f"❌ Failed to analyze {model_name}: {e}")
    
    # Summary comparison
    if results:
        print(f"\n{'='*80}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        df = pd.DataFrame(results)
        print(df.to_string(index=False, float_format='{:.3f}'.format))
        
        # Interpretation
        print(f"\n📊 INTERPRETATION:")
        print(f"- Alpha < 2: May indicate over-training")
        print(f"- Alpha > 6: May indicate under-training") 
        print(f"- LogNorm: Higher = more trained/structured")
        print(f"- Spectral Norm: Lower = more generalizable")
        
        # Save results
        df.to_csv('hrm_weightwatcher_analysis.csv', index=False)
        print(f"\n💾 Results saved to hrm_weightwatcher_analysis.csv")

def analyze_layer_degradation():
    """Check if fine-tuning degraded specific layers"""
    print(f"\n{'='*60}")
    print("LAYER-BY-LAYER DEGRADATION ANALYSIS")
    print(f"{'='*60}")
    
    try:
        # Compare original vs broken fine-tune
        orig_details, _ = analyze_model('checkpoints/hrm_arc_best.pt', 'Original')
        broken_details, _ = analyze_model('checkpoints/hrm_reasoning_agi1_step_3000.pt', 'Broken')
        
        if orig_details is not None and broken_details is not None:
            print(f"\nLayer Degradation Analysis:")
            print(f"{'Layer':<25} {'Orig Alpha':<10} {'Broken Alpha':<12} {'Change':<8}")
            print("-" * 60)
            
            for i, (orig_row, broken_row) in enumerate(zip(orig_details.iterrows(), broken_details.iterrows())):
                layer = orig_row[1]['layer_id']
                orig_alpha = orig_row[1].get('alpha', 0)
                broken_alpha = broken_row[1].get('alpha', 0)
                change = broken_alpha - orig_alpha
                
                status = "⚠️" if abs(change) > 1.0 else "✓"
                print(f"{layer:<25} {orig_alpha:<10.3f} {broken_alpha:<12.3f} {change:+8.3f} {status}")
                
    except Exception as e:
        print(f"❌ Layer analysis failed: {e}")

if __name__ == "__main__":
    compare_models()
    analyze_layer_degradation()