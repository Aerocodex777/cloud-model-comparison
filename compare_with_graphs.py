"""
TGAT vs TGN - Comparison with Visual Output
Generates graphs and metrics visualization
"""

import torch
from tgat_model import TGAT
from tgn_model import TGN
from comparison_framework import PerformanceProfiler, TemporalGraphDataGenerator
from visualizer import MetricsVisualizer
import os


def demo_with_visualizations():
    """Run comparison and generate visualizations"""
    print("\n" + "="*80)
    print("TGAT vs TGN - COMPREHENSIVE COMPARISON WITH VISUALIZATIONS")
    print("="*80 + "\n")
    
    # Step 1: Generate data
    print("[1/6] Generating temporal graph data...")
    gen = TemporalGraphDataGenerator()
    data = gen.generate_temporal_graph(
        num_nodes=300,
        num_edges=1000,
        num_timestamps=20,
        feature_dim=16
    )
    print("✓ Generated: 300 nodes, 1000 edges\n")
    
    # Step 2: Create models
    print("[2/6] Creating models...")
    tgat = TGAT(
        in_channels=16,
        hidden_channels=32,
        out_channels=16,
        num_layers=2,
        heads=4
    )
    
    tgn = TGN(
        in_channels=16,
        hidden_channels=32,
        out_channels=16,
        num_layers=2,
        heads=4
    )
    print("✓ Created TGAT and TGN models\n")
    
    # Step 3: Initialize metrics
    metrics = {
        'tgat': {},
        'tgn': {}
    }
    
    # Step 4: Compare model sizes
    print("[3/6] Comparing model sizes...")
    profiler = PerformanceProfiler()
    
    tgat_size = profiler.measure_model_size(tgat)
    tgn_size = profiler.measure_model_size(tgn)
    
    metrics['tgat'].update(tgat_size)
    metrics['tgn'].update(tgn_size)
    
    print(f"  TGAT: {tgat_size['total_parameters']:,} parameters ({tgat_size['model_size_mb']:.4f} MB)")
    print(f"  TGN:  {tgn_size['total_parameters']:,} parameters ({tgn_size['model_size_mb']:.4f} MB)")
    
    param_diff = ((tgat_size['total_parameters'] - tgn_size['total_parameters']) 
                  / tgn_size['total_parameters'] * 100)
    print(f"  → TGAT is {abs(param_diff):.1f}% {'smaller' if param_diff < 0 else 'larger'}\n")
    
    # Step 5: Compare inference
    print("[4/6] Testing inference speed...")
    
    tgat_time = profiler.measure_inference_time(tgat, data, num_runs=30)
    tgn_time = profiler.measure_inference_time(tgn, data, num_runs=30)
    
    metrics['tgat']['inference_time'] = tgat_time['mean_time_ms']
    metrics['tgn']['inference_time'] = tgn_time['mean_time_ms']
    
    print(f"  TGAT: {tgat_time['mean_time_ms']:.2f} ± {tgat_time['std_time_ms']:.2f} ms")
    print(f"  TGN:  {tgn_time['mean_time_ms']:.2f} ± {tgn_time['std_time_ms']:.2f} ms")
    
    speedup = tgn_time['mean_time_ms'] / tgat_time['mean_time_ms']
    faster = 'TGAT' if speedup > 1 else 'TGN'
    print(f"  → {faster} is {abs(speedup):.2f}x faster\n")
    
    # Step 6: Compare memory
    print("[5/6] Measuring memory usage...")
    
    tgat_mem = profiler.measure_memory_usage(tgat, data)
    tgn_mem = profiler.measure_memory_usage(tgn, data)
    
    metrics['tgat']['memory_usage'] = tgat_mem['memory_used_mb']
    metrics['tgn']['memory_usage'] = tgn_mem['memory_used_mb']
    
    print(f"  TGAT: {tgat_mem['memory_used_mb']:.2f} MB")
    print(f"  TGN:  {tgn_mem['memory_used_mb']:.2f} MB\n")
    
    # Step 7: Generate visualizations
    print("[6/6] Generating visual reports...")
    viz = MetricsVisualizer()
    
    output_files = viz.create_all_visualizations(metrics, 'comparison_outputs')
    
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80 + "\n")
    
    print("📊 METRICS GENERATED:")
    print(f"  ✓ Model Size:        TGAT={tgat_size['total_parameters']:,} vs TGN={tgn_size['total_parameters']:,}")
    print(f"  ✓ Inference Speed:   TGAT={tgat_time['mean_time_ms']:.2f}ms vs TGN={tgn_time['mean_time_ms']:.2f}ms")
    print(f"  ✓ Memory Usage:      TGAT={tgat_mem['memory_used_mb']:.2f}MB vs TGN={tgn_mem['memory_used_mb']:.2f}MB")
    print(f"  ✓ Speedup Factor:    {speedup:.2f}x")
    
    print("\n📈 VISUALIZATIONS CREATED:")
    for i, file in enumerate(output_files, 1):
        print(f"  {i}. {file}")
    
    print("\n🎯 RECOMMENDATIONS:")
    if param_diff < -50:
        print("  → TGAT is significantly smaller - better for deployment")
    elif param_diff > 50:
        print("  → TGN is significantly smaller - better for deployment")
    else:
        print("  → Both models have comparable sizes")
    
    if speedup > 2:
        print(f"  → TGAT is {speedup:.1f}x faster - ideal for real-time applications")
    elif speedup < 0.5:
        print(f"  → TGN is {1/speedup:.1f}x faster - ideal for real-time applications")
    else:
        print("  → Both models have similar inference speed")
    
    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETE!")
    print("="*80 + "\n")
    
    print(f"📁 All visualizations saved to: comparison_outputs/")
    print(f"   Open the PNG files to view detailed graphs and metrics!")
    print("\n")


if __name__ == "__main__":
    try:
        demo_with_visualizations()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
