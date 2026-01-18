"""
Comprehensive Temporal Graph Model Comparison
Compares TGAT, TGN, DyRep, EvolveGCN, JODIE, and CTDG models
Generates detailed metrics and visualizations
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import time
import traceback

from comparison_framework import PerformanceProfiler, TemporalGraphDataGenerator
from tgat_model import TGAT
from tgn_model import TGN
from dyrep_model import DyRep
from evolvegcn_model import EvolveGCN
from jodie_model import JODIE
from ctdg_model import CTDG


class ComprehensiveComparison:
    """Comprehensive comparison of all temporal graph models"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.profiler = PerformanceProfiler()
        self.metrics = {}
        self.models = {}
        
        # Create output directory
        self.output_dir = 'comparison_outputs'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def initialize_models(self, config):
        """Initialize all models with the same configuration"""
        print("\n[1/7] Initializing models...")
        
        model_configs = {
            'TGAT': TGAT,
            'TGN': TGN,
            'DyRep': DyRep,
            'EvolveGCN': EvolveGCN,
            'JODIE': JODIE,
            'CTDG': CTDG
        }
        
        for name, model_class in model_configs.items():
            try:
                # TGN uses 'heads' instead of 'num_heads'
                if name == 'TGN':
                    model = model_class(
                        in_channels=config['in_channels'],
                        hidden_channels=config['hidden_channels'],
                        out_channels=config['out_channels'],
                        num_layers=config['num_layers'],
                        heads=config['num_heads'],
                        dropout=config['dropout']
                    ).to(self.device)
                else:
                    model = model_class(
                        in_channels=config['in_channels'],
                        hidden_channels=config['hidden_channels'],
                        out_channels=config['out_channels'],
                        num_layers=config['num_layers'],
                        num_heads=config['num_heads'],
                        dropout=config['dropout']
                    ).to(self.device)
                
                self.models[name] = model
                self.metrics[name] = {}
                print(f"  ✓ {name} initialized")
            except Exception as e:
                print(f"  ✗ {name} failed: {e}")
                traceback.print_exc()
        
        print(f"\n  Successfully initialized {len(self.models)}/6 models\n")
    
    def generate_data(self, config):
        """Generate temporal graph data"""
        print("[2/7] Generating temporal graph data...")
        
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
                data[key] = data[key].to(self.device)
        
        print(f"  ✓ Generated: {config['num_nodes']} nodes, {config['num_edges']} edges\n")
        return data
    
    def compare_model_sizes(self):
        """Compare model architecture sizes"""
        print("[3/7] Comparing model sizes...")
        
        for name, model in self.models.items():
            size_metrics = self.profiler.measure_model_size(model)
            self.metrics[name]['total_parameters'] = size_metrics['total_parameters']
            self.metrics[name]['trainable_parameters'] = size_metrics['trainable_parameters']
            self.metrics[name]['model_size_mb'] = size_metrics['model_size_mb']
            
            print(f"  {name:12s}: {size_metrics['total_parameters']:>10,} params, "
                  f"{size_metrics['model_size_mb']:>8.2f} MB")
        
        print()
    
    def compare_inference_speed(self, data):
        """Compare inference speed"""
        print("[4/7] Comparing inference speed...")
        
        inputs = {
            'x': data['x'],
            'edge_index': data['edge_index'],
            'timestamps': data['timestamps']
        }
        
        for name, model in self.models.items():
            try:
                model.eval()
                
                def forward_fn(inputs):
                    with torch.no_grad():
                        return model(inputs['x'], inputs['edge_index'], inputs['timestamps'])
                
                time_metrics = self.profiler.measure_inference_time(forward_fn, inputs, num_runs=50)
                self.metrics[name]['inference_time_ms'] = time_metrics['mean_time_ms']
                self.metrics[name]['inference_std_ms'] = time_metrics['std_time_ms']
                
                print(f"  {name:12s}: {time_metrics['mean_time_ms']:>8.2f} ± "
                      f"{time_metrics['std_time_ms']:>6.2f} ms")
            except Exception as e:
                print(f"  {name:12s}: Failed - {e}")
                self.metrics[name]['inference_time_ms'] = float('inf')
                self.metrics[name]['inference_std_ms'] = 0
        
        print()
    
    def compare_memory_usage(self, data):
        """Compare memory usage"""
        print("[5/7] Comparing memory usage...")
        
        inputs = {
            'x': data['x'],
            'edge_index': data['edge_index'],
            'timestamps': data['timestamps']
        }
        
        for name, model in self.models.items():
            try:
                def forward_fn(inputs):
                    return model(inputs['x'], inputs['edge_index'], inputs['timestamps'])
                
                mem_metrics = self.profiler.measure_memory_usage(forward_fn, inputs)
                self.metrics[name]['memory_usage_mb'] = mem_metrics['memory_used_mb']
                
                print(f"  {name:12s}: {mem_metrics['memory_used_mb']:>8.2f} MB")
            except Exception as e:
                print(f"  {name:12s}: Failed - {e}")
                self.metrics[name]['memory_usage_mb'] = 0
        
        print()
    
    def compare_training_performance(self, data, epochs=5):
        """Compare training performance"""
        print(f"[6/7] Comparing training performance ({epochs} epochs)...")
        
        criterion = nn.CrossEntropyLoss()
        
        for name, model in self.models.items():
            try:
                model.train()
                optimizer = optim.Adam(model.parameters(), lr=0.01)
                
                start_time = time.time()
                
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    
                    output = model(data['x'], data['edge_index'], data['timestamps'])
                    loss = criterion(output, data['y'])
                    
                    loss.backward()
                    optimizer.step()
                
                training_time = time.time() - start_time
                self.metrics[name]['training_time_s'] = training_time
                self.metrics[name]['time_per_epoch_s'] = training_time / epochs
                
                print(f"  {name:12s}: {training_time:>8.2f}s total, "
                      f"{training_time/epochs:>6.2f}s/epoch")
            except Exception as e:
                print(f"  {name:12s}: Failed - {e}")
                self.metrics[name]['training_time_s'] = float('inf')
                self.metrics[name]['time_per_epoch_s'] = float('inf')
        
        print()
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        print("[7/7] Generating visualizations...")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
        
        # 1. Model Size Comparison
        self._plot_model_sizes()
        
        # 2. Inference Speed Comparison
        self._plot_inference_speed()
        
        # 3. Memory Usage Comparison
        self._plot_memory_usage()
        
        # 4. Training Performance Comparison
        self._plot_training_performance()
        
        # 5. Comprehensive Metrics Table
        self._plot_metrics_table()
        
        # 6. Radar Chart
        self._plot_radar_chart()
        
        # 7. Efficiency Analysis
        self._plot_efficiency_analysis()
        
        print(f"\n  ✓ All visualizations saved to {self.output_dir}/\n")
    
    def _plot_model_sizes(self):
        """Plot model size comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        models = list(self.metrics.keys())
        params = [self.metrics[m]['total_parameters'] for m in models]
        sizes = [self.metrics[m]['model_size_mb'] for m in models]
        
        # Parameters bar chart
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        bars1 = ax1.bar(models, params, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Number of Parameters', fontsize=12, fontweight='bold')
        ax1.set_title('Model Parameters Comparison', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=9)
        
        # Model size bar chart
        bars2 = ax2.bar(models, sizes, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Model Size (MB)', fontsize=12, fontweight='bold')
        ax2.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/model_sizes.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_inference_speed(self):
        """Plot inference speed comparison"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(self.metrics.keys())
        times = [self.metrics[m]['inference_time_ms'] for m in models]
        stds = [self.metrics[m]['inference_std_ms'] for m in models]
        
        # Filter out failed models
        valid_data = [(m, t, s) for m, t, s in zip(models, times, stds) if t != float('inf')]
        if valid_data:
            models, times, stds = zip(*valid_data)
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(models)))
        bars = ax.bar(models, times, yerr=stds, color=colors, 
                     edgecolor='black', linewidth=1.5, capsize=5)
        
        ax.set_ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Inference Speed Comparison (Lower is Better)', 
                    fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, std in zip(bars, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/inference_speed.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_usage(self):
        """Plot memory usage comparison"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(self.metrics.keys())
        memory = [self.metrics[m]['memory_usage_mb'] for m in models]
        
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(models)))
        bars = ax.bar(models, memory, color=colors, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
        ax.set_title('Memory Usage Comparison (Lower is Better)', 
                    fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/memory_usage.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_training_performance(self):
        """Plot training performance comparison"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(self.metrics.keys())
        times = [self.metrics[m]['training_time_s'] for m in models]
        
        # Filter out failed models
        valid_data = [(m, t) for m, t in zip(models, times) if t != float('inf')]
        if valid_data:
            models, times = zip(*valid_data)
        
        colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(models)))
        bars = ax.bar(models, times, color=colors, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Training Performance Comparison (5 epochs)', 
                    fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}s',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/training_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_table(self):
        """Create comprehensive metrics table"""
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data
        models = list(self.metrics.keys())
        table_data = []
        
        for model in models:
            m = self.metrics[model]
            row = [
                model,
                f"{m['total_parameters']:,}",
                f"{m['model_size_mb']:.2f} MB",
                f"{m['inference_time_ms']:.2f} ms" if m['inference_time_ms'] != float('inf') else 'N/A',
                f"{m['memory_usage_mb']:.2f} MB",
                f"{m['training_time_s']:.2f}s" if m['training_time_s'] != float('inf') else 'N/A'
            ]
            table_data.append(row)
        
        headers = ['Model', 'Parameters', 'Model Size', 'Inference Time', 
                  'Memory Usage', 'Training Time']
        
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=[0.12, 0.18, 0.15, 0.18, 0.18, 0.18])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(len(headers)):
            cell = table[(0, i)]
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white', fontsize=12)
        
        # Style rows with alternating colors
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#f0f0f0')
                else:
                    cell.set_facecolor('#ffffff')
                cell.set_edgecolor('#cccccc')
        
        plt.title('Comprehensive Model Comparison Metrics', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.savefig(f'{self.output_dir}/metrics_table.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_radar_chart(self):
        """Create radar chart for normalized metrics"""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        models = list(self.metrics.keys())
        
        # Normalize metrics (lower is better, so we invert)
        categories = ['Speed', 'Memory', 'Size', 'Training']
        
        # Get values and normalize
        def normalize(values):
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                return [0.5] * len(values)
            # Invert so higher is better
            return [1 - (v - min_val) / (max_val - min_val) for v in values]
        
        inference_times = [self.metrics[m]['inference_time_ms'] for m in models 
                          if self.metrics[m]['inference_time_ms'] != float('inf')]
        memory_usage = [self.metrics[m]['memory_usage_mb'] for m in models]
        model_sizes = [self.metrics[m]['model_size_mb'] for m in models]
        training_times = [self.metrics[m]['training_time_s'] for m in models 
                         if self.metrics[m]['training_time_s'] != float('inf')]
        
        # Create angles for radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        for idx, model in enumerate(models):
            m = self.metrics[model]
            
            # Skip if critical metrics are missing
            if m['inference_time_ms'] == float('inf') or m['training_time_s'] == float('inf'):
                continue
            
            values = [
                1 - (m['inference_time_ms'] - min(inference_times)) / (max(inference_times) - min(inference_times)) if len(inference_times) > 1 else 0.5,
                1 - (m['memory_usage_mb'] - min(memory_usage)) / (max(memory_usage) - min(memory_usage)) if len(memory_usage) > 1 else 0.5,
                1 - (m['model_size_mb'] - min(model_sizes)) / (max(model_sizes) - min(model_sizes)) if len(model_sizes) > 1 else 0.5,
                1 - (m['training_time_s'] - min(training_times)) / (max(training_times) - min(training_times)) if len(training_times) > 1 else 0.5
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        plt.title('Model Performance Radar Chart\n(Higher is Better)', 
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.savefig(f'{self.output_dir}/radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_efficiency_analysis(self):
        """Plot efficiency analysis (speed vs size tradeoff)"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        models = list(self.metrics.keys())
        
        # Prepare data
        x_data = []  # Model size
        y_data = []  # Inference time
        labels = []
        
        for model in models:
            m = self.metrics[model]
            if m['inference_time_ms'] != float('inf'):
                x_data.append(m['model_size_mb'])
                y_data.append(m['inference_time_ms'])
                labels.append(model)
        
        # Create scatter plot
        colors = plt.cm.rainbow(np.linspace(0, 1, len(labels)))
        scatter = ax.scatter(x_data, y_data, s=300, c=colors, 
                           alpha=0.6, edgecolors='black', linewidth=2)
        
        # Add labels
        for i, label in enumerate(labels):
            ax.annotate(label, (x_data[i], y_data[i]), 
                       fontsize=12, fontweight='bold',
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        ax.set_xlabel('Model Size (MB)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Inference Time (ms)', fontsize=13, fontweight='bold')
        ax.set_title('Efficiency Analysis: Speed vs Size Tradeoff\n(Lower-Left is Better)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add quadrant lines
        if x_data and y_data:
            median_x = np.median(x_data)
            median_y = np.median(y_data)
            ax.axvline(median_x, color='red', linestyle='--', alpha=0.5, linewidth=2)
            ax.axhline(median_y, color='red', linestyle='--', alpha=0.5, linewidth=2)
            
            # Add quadrant labels
            ax.text(ax.get_xlim()[0] + 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                   ax.get_ylim()[1] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                   'Small & Slow', fontsize=10, style='italic', alpha=0.6)
            ax.text(ax.get_xlim()[1] - 0.15 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                   ax.get_ylim()[1] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                   'Large & Slow', fontsize=10, style='italic', alpha=0.6)
            ax.text(ax.get_xlim()[0] + 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                   ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                   'Small & Fast ✓', fontsize=10, style='italic', alpha=0.6, 
                   color='green', fontweight='bold')
            ax.text(ax.get_xlim()[1] - 0.15 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                   ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                   'Large & Fast', fontsize=10, style='italic', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "="*80)
        print("COMPREHENSIVE COMPARISON SUMMARY")
        print("="*80 + "\n")
        
        # Find best models for each metric
        models = list(self.metrics.keys())
        
        # Best by parameters (smallest)
        params = {m: self.metrics[m]['total_parameters'] for m in models}
        best_params = min(params, key=params.get)
        
        # Best by inference speed (fastest)
        inf_times = {m: self.metrics[m]['inference_time_ms'] for m in models 
                    if self.metrics[m]['inference_time_ms'] != float('inf')}
        best_inference = min(inf_times, key=inf_times.get) if inf_times else 'N/A'
        
        # Best by memory (lowest)
        memory = {m: self.metrics[m]['memory_usage_mb'] for m in models}
        best_memory = min(memory, key=memory.get)
        
        # Best by training (fastest)
        train_times = {m: self.metrics[m]['training_time_s'] for m in models 
                      if self.metrics[m]['training_time_s'] != float('inf')}
        best_training = min(train_times, key=train_times.get) if train_times else 'N/A'
        
        print("🏆 BEST PERFORMERS:")
        print(f"  • Smallest Model:     {best_params} ({params[best_params]:,} parameters)")
        if best_inference != 'N/A':
            print(f"  • Fastest Inference:  {best_inference} ({inf_times[best_inference]:.2f} ms)")
        print(f"  • Lowest Memory:      {best_memory} ({memory[best_memory]:.2f} MB)")
        if best_training != 'N/A':
            print(f"  • Fastest Training:   {best_training} ({train_times[best_training]:.2f}s)")
        
        print("\n📊 DETAILED METRICS:")
        for model in models:
            m = self.metrics[model]
            print(f"\n  {model}:")
            print(f"    Parameters:     {m['total_parameters']:>12,}")
            print(f"    Model Size:     {m['model_size_mb']:>12.2f} MB")
            inf_time = f"{m['inference_time_ms']:.2f} ms" if m['inference_time_ms'] != float('inf') else 'N/A'
            print(f"    Inference:      {inf_time:>12s}")
            print(f"    Memory:         {m['memory_usage_mb']:>12.2f} MB")
            train_time = f"{m['training_time_s']:.2f}s" if m['training_time_s'] != float('inf') else 'N/A'
            print(f"    Training:       {train_time:>12s}")
        
        print("\n" + "="*80)
        print("✅ COMPARISON COMPLETE!")
        print("="*80 + "\n")
        
        print(f"📁 All visualizations saved to: {self.output_dir}/")
        print("   Generated files:")
        print("   1. model_sizes.png - Model parameter and size comparison")
        print("   2. inference_speed.png - Inference speed comparison")
        print("   3. memory_usage.png - Memory usage comparison")
        print("   4. training_performance.png - Training time comparison")
        print("   5. metrics_table.png - Comprehensive metrics table")
        print("   6. radar_chart.png - Multi-dimensional performance radar")
        print("   7. efficiency_analysis.png - Speed vs size tradeoff analysis")
        print()
    
    def run_comparison(self):
        """Run complete comparison"""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEMPORAL GRAPH MODEL COMPARISON")
        print("Models: TGAT, TGN, DyRep, EvolveGCN, JODIE, CTDG")
        print("="*80)
        
        # Configuration
        config = {
            'num_nodes': 500,
            'num_edges': 2000,
            'num_timestamps': 30,
            'in_channels': 32,
            'hidden_channels': 64,
            'out_channels': 16,
            'num_layers': 2,
            'num_heads': 4,
            'dropout': 0.1
        }
        
        # Run comparison steps
        self.initialize_models(config)
        data = self.generate_data(config)
        self.compare_model_sizes()
        self.compare_inference_speed(data)
        self.compare_memory_usage(data)
        self.compare_training_performance(data, epochs=5)
        self.generate_visualizations()
        self.print_summary()


def main():
    """Main entry point"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    comparison = ComprehensiveComparison(device=device)
    comparison.run_comparison()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
