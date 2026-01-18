"""
Quick Start Script - TGAT vs TGA Comparison
Run this to see everything in action
"""

import torch
from tgat_model import TGAT
from tga_model import TGA
from comparison_framework import PerformanceProfiler, TemporalGraphDataGenerator


def quick_demo():
    """Run quick demonstration"""
    print("\n" + "="*80)
    print("TGAT vs TGA - QUICK START DEMO")
    print("="*80 + "\n")
    
    # Generate data
    print("[1] Generating temporal graph data...")
    gen = TemporalGraphDataGenerator()
    data = gen.generate_temporal_graph(
        num_nodes=300,
        num_edges=1000,
        num_timestamps=20,
        feature_dim=16
    )
    print("✓ Generated graph with 300 nodes and 1000 edges\n")
    
    # Create models
    print("[2] Creating models...")
    tgat = TGAT(
        in_channels=16,
        hidden_channels=32,
        out_channels=16,
        num_layers=2,
        heads=4
    )
    
    tga = TGA(
        in_channels=16,
        hidden_channels=32,
        out_channels=16,
        num_layers=2,
        heads=4
    )
    print("✓ Created TGAT and TGA models\n")
    
    # Compare model sizes
    print("[3] Comparing model sizes...")
    profiler = PerformanceProfiler()
    
    tgat_size = profiler.measure_model_size(tgat)
    tga_size = profiler.measure_model_size(tga)
    
    print(f"  TGAT: {tgat_size['total_parameters']:,} parameters ({tgat_size['model_size_mb']:.2f} MB)")
    print(f"  TGA:  {tga_size['total_parameters']:,} parameters ({tga_size['model_size_mb']:.2f} MB)")
    
    param_diff = (tgat_size['total_parameters'] - tga_size['total_parameters']) / tga_size['total_parameters'] * 100
    print(f"  → TGAT is {abs(param_diff):.1f}% {'larger' if param_diff > 0 else 'smaller'}\n")
    
    # Test inference
    print("[4] Testing inference...")
    
    tgat_time = profiler.measure_inference_time(tgat, data, num_runs=20)
    tga_time = profiler.measure_inference_time(tga, data, num_runs=20)
    
    print(f"  TGAT: {tgat_time['mean_time_ms']:.2f} ± {tgat_time['std_time_ms']:.2f} ms")
    print(f"  TGA:  {tga_time['mean_time_ms']:.2f} ± {tga_time['std_time_ms']:.2f} ms")
    
    speedup = tga_time['mean_time_ms'] / tgat_time['mean_time_ms']
    faster = 'TGAT' if speedup > 1 else 'TGA'
    print(f"  → {faster} is {abs(speedup):.2f}x faster\n")
    
    # Test memory usage
    print("[5] Testing memory usage...")
    tgat_mem = profiler.measure_memory_usage(tgat, data)
    tga_mem = profiler.measure_memory_usage(tga, data)
    
    print(f"  TGAT: {tgat_mem['memory_used_mb']:.4f} MB")
    print(f"  TGA:  {tga_mem['memory_used_mb']:.4f} MB")
    
    if tga_mem['memory_used_mb'] > 1e-6:
        mem_diff = (tgat_mem['memory_used_mb'] - tga_mem['memory_used_mb']) / tga_mem['memory_used_mb'] * 100
    else:
        mem_diff = 0
    print(f"  → TGAT uses {abs(mem_diff):.1f}% {'more' if mem_diff > 0 else 'less'} memory\n")
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print("\n✓ Model Comparison Complete!")
    print("\nKey Findings:")
    print(f"  • TGAT is {abs(param_diff):.1f}% {'larger' if param_diff > 0 else 'smaller'} in model size")
    print(f"  • {faster} is {abs(speedup):.2f}x faster in inference")
    print(f"  • TGAT uses {abs(mem_diff):.1f}% {'more' if mem_diff > 0 else 'less'} memory")
    
    print("\nRecommendations:")
    if param_diff > 20:
        print("  → Use TGA if you need a lightweight model")
    else:
        print("  → Both models have comparable sizes")
    
    if speedup > 1.2:
        print("  → TGAT is faster for inference")
    elif speedup < 0.8:
        print("  → TGA is faster for inference")
    else:
        print("  → Both models have similar inference speed")
    
    if mem_diff > 20:
        print("  → TGA is more memory-efficient")
    else:
        print("  → Both models have comparable memory usage")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    try:
        quick_demo()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
