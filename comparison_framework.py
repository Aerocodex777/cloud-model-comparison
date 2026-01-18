"""
Comprehensive TGAT vs TGA Comparison Framework
Compares Temporal Graph Attention Network with Temporal Graph Attention
on multiple metrics: accuracy, speed, memory, scalability, and robustness
"""

import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import os
from typing import Dict, List, Tuple
import json
from datetime import datetime

class ComparisonMetrics:
    """Stores and manages comparison metrics"""
    def __init__(self):
        self.metrics = {
            'tgat': {},
            'tga': {}
        }
    
    def add_metric(self, model_name: str, metric_name: str, value):
        """Add a metric for a model"""
        if model_name not in self.metrics:
            self.metrics[model_name] = {}
        self.metrics[model_name][metric_name] = value
    
    def get_metrics(self, model_name: str) -> Dict:
        """Get all metrics for a model"""
        return self.metrics.get(model_name, {})
    
    def compare(self) -> Dict:
        """Compare metrics between models"""
        comparison = {}
        tgat_metrics = self.metrics.get('tgat', {})
        tga_metrics = self.metrics.get('tga', {})
        
        for key in tgat_metrics.keys():
            if key in tga_metrics:
                tgat_val = tgat_metrics[key]
                tga_val = tga_metrics[key]
                
                if isinstance(tgat_val, (int, float)) and isinstance(tga_val, (int, float)):
                    # Calculate percentage difference
                    if tga_val != 0:
                        diff_percent = ((tgat_val - tga_val) / tga_val) * 100
                    else:
                        diff_percent = 0
                    
                    comparison[key] = {
                        'tgat': tgat_val,
                        'tga': tga_val,
                        'difference_percent': diff_percent,
                        'winner': 'TGAT' if (key in ['accuracy', 'auc'] and tgat_val > tga_val) or 
                                           (key in ['inference_time', 'memory_usage', 'training_time'] and tgat_val < tga_val)
                                else 'TGA'
                    }
        
        return comparison
    
    def save_results(self, filename: str = 'comparison_results.json'):
        """Save results to JSON file"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'tgat': self.metrics['tgat'],
            'tga': self.metrics['tga'],
            'comparison': self.compare()
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return filename


class PerformanceProfiler:
    """Profiles model performance metrics"""
    
    @staticmethod
    def measure_inference_time(model, inputs, num_runs: int = 100) -> Dict:
        """Measure inference time"""
        model.eval()
        times = []
        
        def run_forward():
            with torch.no_grad():
                _ = model(inputs['x'], inputs['edge_index'], inputs['timestamps'])
        
        # Warmup
        for _ in range(10):
            run_forward()
        
        # Measure
        for _ in range(num_runs):
            start = time.time()
            run_forward()
            times.append(time.time() - start)
        
        times = np.array(times[5:])  # Remove first few runs
        return {
            'mean_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000,
        }
    
    @staticmethod
    def measure_memory_usage(model, inputs) -> Dict:
        """Measure memory usage"""
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        process = psutil.Process(os.getpid())
        mem_start = process.memory_info().rss / 1024 / 1024  # MB
        
        model.eval()
        with torch.no_grad():
            output = model(inputs['x'], inputs['edge_index'], inputs['timestamps'])
        
        mem_end = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'memory_used_mb': mem_end - mem_start,
            'total_memory_mb': mem_end,
        }
    
    @staticmethod
    def measure_training_time(model, criterion, optimizer, train_loader, epochs: int = 10) -> Dict:
        """Measure training time"""
        times = []
        
        for epoch in range(epochs):
            start = time.time()
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            times.append(time.time() - start)
        
        times = np.array(times)
        return {
            'total_training_time_s': np.sum(times),
            'mean_epoch_time_s': np.mean(times),
            'min_epoch_time_s': np.min(times),
            'max_epoch_time_s': np.max(times),
        }
    
    @staticmethod
    def measure_model_size(model) -> Dict:
        """Measure model size in parameters and bytes"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        # Estimate model size in MB
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        
        return {
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'non_trainable_parameters': int(non_trainable_params),
            'model_size_mb': model_size_mb,
        }


class TemporalGraphDataGenerator:
    """Generates synthetic temporal graph data for testing"""
    
    @staticmethod
    def generate_temporal_graph(num_nodes: int = 1000, 
                               num_edges: int = 5000,
                               num_timestamps: int = 50,
                               feature_dim: int = 64) -> Dict:
        """Generate synthetic temporal graph data"""
        
        # Edge indices (source, target)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        # Timestamps
        timestamps = torch.randint(0, num_timestamps, (num_edges,))
        
        # Node features
        x = torch.randn(num_nodes, feature_dim)
        
        # Edge features
        edge_attr = torch.randn(num_edges, feature_dim // 2)
        
        # Labels (for classification tasks)
        y = torch.randint(0, 10, (num_nodes,))
        
        return {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'timestamps': timestamps,
            'y': y,
        }


def print_comparison_results(metrics: ComparisonMetrics):
    """Pretty print comparison results"""
    comparison = metrics.compare()
    
    print("\n" + "="*80)
    print("TGAT vs TGA COMPREHENSIVE COMPARISON RESULTS")
    print("="*80 + "\n")
    
    for metric_name, values in comparison.items():
        print(f"\n{metric_name.upper()}")
        print("-" * 40)
        print(f"  TGAT:              {values['tgat']:.4f}" if isinstance(values['tgat'], float) else f"  TGAT:              {values['tgat']}")
        print(f"  TGA:               {values['tga']:.4f}" if isinstance(values['tga'], float) else f"  TGA:               {values['tga']}")
        print(f"  Difference:        {values['difference_percent']:.2f}%")
        print(f"  Winner:            {values['winner']}")


if __name__ == "__main__":
    print("Comparison Framework Loaded Successfully!")
    print("Available Classes:")
    print("  - ComparisonMetrics: Manage and compare metrics")
    print("  - PerformanceProfiler: Profile model performance")
    print("  - TemporalGraphDataGenerator: Generate test data")
    print("\nUse: from comparison_framework import *")
