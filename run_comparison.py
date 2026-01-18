"""
Comprehensive TGAT vs TGA Comparison Script
Runs all comparisons and generates detailed reports
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import traceback

from comparison_framework import (
    ComparisonMetrics, PerformanceProfiler, TemporalGraphDataGenerator,
    print_comparison_results
)
from tgat_model import TGAT
from tga_model import TGA


class ComparisonRunner:
    """Orchestrates the comparison between TGAT and TGA"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.metrics = ComparisonMetrics()
        self.profiler = PerformanceProfiler()
    
    def run_full_comparison(self):
        """Run complete comparison suite"""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE TGAT vs TGA COMPARISON")
        print("="*80 + "\n")
        
        # Configuration
        config = {
            'num_nodes': 500,
            'num_edges': 2000,
            'num_timestamps': 30,
            'in_channels': 32,
            'hidden_channels': 64,
            'out_channels': 16,
            'num_layers': 2,
            'num_heads': 8,
            'dropout': 0.1,
            'device': self.device
        }
        
        try:
            print("[1/6] Generating Temporal Graph Data...")
            data = self._generate_data(config)
            print("✓ Data generation complete\n")
        except Exception as e:
            print(f"✗ Data generation failed: {e}\n")
            traceback.print_exc()
            return
        
        try:
            print("[2/6] Comparing Model Sizes...")
            self._compare_model_sizes(config)
            print("✓ Model size comparison complete\n")
        except Exception as e:
            print(f"✗ Model size comparison failed: {e}\n")
            traceback.print_exc()
        
        try:
            print("[3/6] Comparing Inference Speed...")
            self._compare_inference_speed(config, data)
            print("✓ Inference speed comparison complete\n")
        except Exception as e:
            print(f"✗ Inference speed comparison failed: {e}\n")
            traceback.print_exc()
        
        try:
            print("[4/6] Comparing Memory Usage...")
            self._compare_memory_usage(config, data)
            print("✓ Memory usage comparison complete\n")
        except Exception as e:
            print(f"✗ Memory usage comparison failed: {e}\n")
            traceback.print_exc()
        
        try:
            print("[5/6] Comparing Training Performance...")
            self._compare_training(config, data)
            print("✓ Training performance comparison complete\n")
        except Exception as e:
            print(f"✗ Training comparison failed: {e}\n")
            traceback.print_exc()
        
        try:
            print("[6/6] Generating Final Report...")
            output_file = self._generate_report()
            print(f"✓ Report saved to: {output_file}\n")
        except Exception as e:
            print(f"✗ Report generation failed: {e}\n")
            traceback.print_exc()
        
        # Print summary
        print_comparison_results(self.metrics)
    
    def _generate_data(self, config):
        """Generate temporal graph data"""
        gen = TemporalGraphDataGenerator()
        data = gen.generate_temporal_graph(
            num_nodes=config['num_nodes'],
            num_edges=config['num_edges'],
            num_timestamps=config['num_timestamps'],
            feature_dim=config['in_channels']
        )
        
        # Move to device
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(config['device'])
        
        return data
    
    def _compare_model_sizes(self, config):
        """Compare model architecture sizes"""
        print("\n  Model Size Comparison:")
        print("  " + "-"*60)
        
        # TGAT
        tgat = TGAT(
            in_channels=config['in_channels'],
            hidden_channels=config['hidden_channels'],
            out_channels=config['out_channels'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        ).to(config['device'])
        
        tgat_metrics = self.profiler.measure_model_size(tgat)
        self.metrics.add_metric('tgat', 'total_parameters', tgat_metrics['total_parameters'])
        self.metrics.add_metric('tgat', 'model_size_mb', tgat_metrics['model_size_mb'])
        
        print(f"  TGAT:")
        print(f"    - Total Parameters: {tgat_metrics['total_parameters']:,}")
        print(f"    - Trainable Parameters: {tgat_metrics['trainable_parameters']:,}")
        print(f"    - Model Size: {tgat_metrics['model_size_mb']:.2f} MB")
        
        # TGA
        tga = TGA(
            in_channels=config['in_channels'],
            hidden_channels=config['hidden_channels'],
            out_channels=config['out_channels'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        ).to(config['device'])
        
        tga_metrics = self.profiler.measure_model_size(tga)
        self.metrics.add_metric('tga', 'total_parameters', tga_metrics['total_parameters'])
        self.metrics.add_metric('tga', 'model_size_mb', tga_metrics['model_size_mb'])
        
        print(f"  TGA:")
        print(f"    - Total Parameters: {tga_metrics['total_parameters']:,}")
        print(f"    - Trainable Parameters: {tga_metrics['trainable_parameters']:,}")
        print(f"    - Model Size: {tga_metrics['model_size_mb']:.2f} MB")
        
        # Calculate difference
        param_diff = ((tgat_metrics['total_parameters'] - tga_metrics['total_parameters']) 
                     / tga_metrics['total_parameters'] * 100)
        print(f"\n  TGAT is {abs(param_diff):.1f}% {'larger' if param_diff > 0 else 'smaller'} than TGA")
    
    def _compare_inference_speed(self, config, data):
        """Compare inference speed"""
        print("\n  Inference Speed Comparison:")
        print("  " + "-"*60)
        
        # Create models
        tgat = TGAT(
            in_channels=config['in_channels'],
            hidden_channels=config['hidden_channels'],
            out_channels=config['out_channels'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        ).to(config['device'])
        
        tga = TGA(
            in_channels=config['in_channels'],
            hidden_channels=config['hidden_channels'],
            out_channels=config['out_channels'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        ).to(config['device'])
        
        # Prepare inference input
        inputs = {
            'x': data['x'],
            'edge_index': data['edge_index'],
            'timestamps': data['timestamps']
        }
        
        # Measure TGAT
        def tgat_forward(inputs):
            return tgat(inputs['x'], inputs['edge_index'], inputs['timestamps'])
        
        tgat_time = self.profiler.measure_inference_time(tgat_forward, inputs, num_runs=50)
        self.metrics.add_metric('tgat', 'inference_time', tgat_time['mean_time_ms'])
        
        print(f"  TGAT:")
        print(f"    - Mean Inference Time: {tgat_time['mean_time_ms']:.2f} ms")
        print(f"    - Std Dev: {tgat_time['std_time_ms']:.2f} ms")
        print(f"    - Min/Max: {tgat_time['min_time_ms']:.2f} / {tgat_time['max_time_ms']:.2f} ms")
        
        # Measure TGA
        def tga_forward(inputs):
            return tga(inputs['x'], inputs['edge_index'], inputs['timestamps'])
        
        tga_time = self.profiler.measure_inference_time(tga_forward, inputs, num_runs=50)
        self.metrics.add_metric('tga', 'inference_time', tga_time['mean_time_ms'])
        
        print(f"  TGA:")
        print(f"    - Mean Inference Time: {tga_time['mean_time_ms']:.2f} ms")
        print(f"    - Std Dev: {tga_time['std_time_ms']:.2f} ms")
        print(f"    - Min/Max: {tga_time['min_time_ms']:.2f} / {tga_time['max_time_ms']:.2f} ms")
        
        # Calculate speedup
        speedup = tga_time['mean_time_ms'] / tgat_time['mean_time_ms']
        faster = 'TGAT' if speedup > 1 else 'TGA'
        print(f"\n  {faster} is {abs(speedup):.2f}x faster than {'TGA' if faster == 'TGAT' else 'TGAT'}")
    
    def _compare_memory_usage(self, config, data):
        """Compare memory usage"""
        print("\n  Memory Usage Comparison:")
        print("  " + "-"*60)
        
        # Create models
        tgat = TGAT(
            in_channels=config['in_channels'],
            hidden_channels=config['hidden_channels'],
            out_channels=config['out_channels'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        ).to(config['device'])
        
        tga = TGA(
            in_channels=config['in_channels'],
            hidden_channels=config['hidden_channels'],
            out_channels=config['out_channels'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        ).to(config['device'])
        
        # Prepare inference input
        inputs = {
            'x': data['x'],
            'edge_index': data['edge_index'],
            'timestamps': data['timestamps']
        }
        
        def tgat_forward(inputs):
            return tgat(inputs['x'], inputs['edge_index'], inputs['timestamps'])
        
        def tga_forward(inputs):
            return tga(inputs['x'], inputs['edge_index'], inputs['timestamps'])
        
        # Measure TGAT
        tgat_mem = self.profiler.measure_memory_usage(tgat_forward, inputs)
        self.metrics.add_metric('tgat', 'memory_usage', tgat_mem['memory_used_mb'])
        
        print(f"  TGAT:")
        print(f"    - Memory Used: {tgat_mem['memory_used_mb']:.2f} MB")
        print(f"    - Total Memory: {tgat_mem['total_memory_mb']:.2f} MB")
        
        # Measure TGA
        tga_mem = self.profiler.measure_memory_usage(tga_forward, inputs)
        self.metrics.add_metric('tga', 'memory_usage', tga_mem['memory_used_mb'])
        
        print(f"  TGA:")
        print(f"    - Memory Used: {tga_mem['memory_used_mb']:.2f} MB")
        print(f"    - Total Memory: {tga_mem['total_memory_mb']:.2f} MB")
        
        # Calculate difference
        mem_diff = ((tgat_mem['memory_used_mb'] - tga_mem['memory_used_mb']) 
                   / tga_mem['memory_used_mb'] * 100)
        print(f"\n  TGAT uses {abs(mem_diff):.1f}% {'more' if mem_diff > 0 else 'less'} memory than TGA")
    
    def _compare_training(self, config, data):
        """Compare training performance"""
        print("\n  Training Performance Comparison:")
        print("  " + "-"*60)
        
        # Create dummy labels
        batch_size = 32
        num_batches = 3
        
        x_list = [data['x'] for _ in range(num_batches)]
        edge_index_list = [data['edge_index'] for _ in range(num_batches)]
        timestamps_list = [data['timestamps'] for _ in range(num_batches)]
        y_list = [data['y'] for _ in range(num_batches)]
        
        print(f"  Training Configuration:")
        print(f"    - Num Batches: {num_batches}")
        print(f"    - Epochs: 5")
        
        # Train TGAT
        tgat = TGAT(
            in_channels=config['in_channels'],
            hidden_channels=config['hidden_channels'],
            out_channels=config['out_channels'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        ).to(config['device'])
        
        tgat_time = self._train_model(tgat, x_list, edge_index_list, timestamps_list, y_list, epochs=5)
        self.metrics.add_metric('tgat', 'training_time', tgat_time)
        
        print(f"  TGAT Training Time: {tgat_time:.2f} seconds")
        
        # Train TGA
        tga = TGA(
            in_channels=config['in_channels'],
            hidden_channels=config['hidden_channels'],
            out_channels=config['out_channels'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        ).to(config['device'])
        
        tga_time = self._train_model(tga, x_list, edge_index_list, timestamps_list, y_list, epochs=5)
        self.metrics.add_metric('tga', 'training_time', tga_time)
        
        print(f"  TGA Training Time: {tga_time:.2f} seconds")
        
        # Calculate speedup
        speedup = tga_time / tgat_time
        faster = 'TGAT' if speedup > 1 else 'TGA'
        print(f"\n  {faster} trains {abs(speedup):.2f}x faster than {'TGA' if faster == 'TGAT' else 'TGAT'}")
    
    def _train_model(self, model, x_list, edge_index_list, timestamps_list, y_list, epochs=5):
        """Train a model and measure time"""
        import time
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            model.train()
            for i in range(len(x_list)):
                optimizer.zero_grad()
                
                logits = model(x_list[i], edge_index_list[i], timestamps_list[i])
                loss = criterion(logits, y_list[i])
                
                loss.backward()
                optimizer.step()
        
        return time.time() - start_time
    
    def _generate_report(self):
        """Generate final comparison report"""
        return self.metrics.save_results('comparison_results.json')


def main():
    """Main entry point"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    runner = ComparisonRunner(device=device)
    runner.run_full_comparison()


if __name__ == "__main__":
    main()
