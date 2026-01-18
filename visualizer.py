"""
Visualization Framework for TGAT vs TGN Comparison
Generates graphs and charts for all metrics
"""

import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import numpy as np
from typing import Dict
import os

# Create output directory
os.makedirs('comparison_outputs', exist_ok=True)


class MetricsVisualizer:
    """Visualize comparison metrics"""
    
    @staticmethod
    def plot_model_comparison(metrics: Dict, output_path: str = 'comparison_outputs'):
        """Create comparison plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('TGAT vs TGN - Comprehensive Comparison', fontsize=16, fontweight='bold')
        
        # 1. Parameter Count Comparison
        ax1 = axes[0, 0]
        models = ['TGAT', 'TGN']
        params = [
            metrics['tgat'].get('total_parameters', 0),
            metrics['tgn'].get('total_parameters', 0)
        ]
        colors = ['#2ecc71', '#e74c3c']
        bars1 = ax1.bar(models, params, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Number of Parameters', fontsize=11, fontweight='bold')
        ax1.set_title('Model Size Comparison', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontweight='bold')
        
        # 2. Inference Speed Comparison
        ax2 = axes[0, 1]
        inference_times = [
            metrics['tgat'].get('inference_time', 0),
            metrics['tgn'].get('inference_time', 0)
        ]
        bars2 = ax2.bar(models, inference_times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Inference Time (ms)', fontsize=11, fontweight='bold')
        ax2.set_title('Inference Speed Comparison', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f} ms',
                    ha='center', va='bottom', fontweight='bold')
        
        # 3. Memory Usage Comparison
        ax3 = axes[1, 0]
        memory_usage = [
            metrics['tgat'].get('memory_usage', 0),
            metrics['tgn'].get('memory_usage', 0)
        ]
        bars3 = ax3.bar(models, memory_usage, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax3.set_ylabel('Memory Usage (MB)', fontsize=11, fontweight='bold')
        ax3.set_title('Memory Consumption Comparison', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        for bar in bars3:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f} MB',
                        ha='center', va='bottom', fontweight='bold')
        
        # 4. Model Size (MB) Comparison
        ax4 = axes[1, 1]
        model_sizes = [
            metrics['tgat'].get('model_size_mb', 0),
            metrics['tgn'].get('model_size_mb', 0)
        ]
        bars4 = ax4.bar(models, model_sizes, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax4.set_ylabel('Model Size (MB)', fontsize=11, fontweight='bold')
        ax4.set_title('Model File Size Comparison', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f} MB',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        filename = os.path.join(output_path, 'comparison_metrics.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
        
        return filename
    
    @staticmethod
    def plot_speedup_analysis(metrics: Dict, output_path: str = 'comparison_outputs'):
        """Create speedup analysis plot"""
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        tgat_time = metrics['tgat'].get('inference_time', 1)
        tgn_time = metrics['tgn'].get('inference_time', 1)
        
        speedup = tgn_time / tgat_time if tgat_time > 0 else 1
        
        # Create speedup bar
        if speedup > 1:
            label = f'TGAT is {speedup:.2f}x FASTER'
            color = '#2ecc71'
        else:
            label = f'TGN is {1/speedup:.2f}x FASTER'
            color = '#e74c3c'
        
        bars = ax.barh(['Speedup Factor'], [speedup], color=color, alpha=0.7, edgecolor='black', linewidth=2, height=0.5)
        
        # Add reference line at 1x
        ax.axvline(x=1, color='black', linestyle='--', linewidth=2, label='Equal Performance')
        
        ax.set_xlabel('Speedup Factor', fontsize=12, fontweight='bold')
        ax.set_title('TGAT vs TGN - Speedup Analysis', fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(speedup * 1.2, 1.2))
        ax.grid(axis='x', alpha=0.3)
        
        # Add text annotation
        ax.text(speedup/2, 0, f'{speedup:.2f}x', ha='center', va='center', 
               fontsize=16, fontweight='bold', color='white')
        ax.text(speedup + 0.1, 0, label, ha='left', va='center', fontsize=12, fontweight='bold')
        
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        filename = os.path.join(output_path, 'speedup_analysis.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
        
        return filename
    
    @staticmethod
    def plot_efficiency_metrics(metrics: Dict, output_path: str = 'comparison_outputs'):
        """Create efficiency metrics plot"""
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Calculate efficiency scores (lower is better)
        tgat_params = metrics['tgat'].get('total_parameters', 1)
        tgn_params = metrics['tgn'].get('total_parameters', 1)
        
        tgat_time = metrics['tgat'].get('inference_time', 1)
        tgn_time = metrics['tgn'].get('inference_time', 1)
        
        # Efficiency = Parameters / Speed (lower is better)
        tgat_efficiency = tgat_params / tgat_time
        tgn_efficiency = tgn_params / tgn_time
        
        models = ['TGAT', 'TGN']
        efficiency_scores = [tgat_efficiency, tgn_efficiency]
        colors = ['#2ecc71', '#e74c3c']
        
        bars = ax.bar(models, efficiency_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        ax.set_ylabel('Efficiency Score (Params/Time)', fontsize=12, fontweight='bold')
        ax.set_title('Model Efficiency Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}',
                   ha='center', va='bottom', fontweight='bold')
        
        # Add efficiency improvement
        improvement = ((tgn_efficiency - tgat_efficiency) / tgn_efficiency) * 100
        ax.text(0.5, max(efficiency_scores) * 0.9, 
               f'TGAT is {abs(improvement):.1f}% {"more" if improvement > 0 else "less"} efficient',
               ha='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        plt.tight_layout()
        filename = os.path.join(output_path, 'efficiency_metrics.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
        
        return filename
    
    @staticmethod
    def plot_summary_table(metrics: Dict, output_path: str = 'comparison_outputs'):
        """Create summary table visualization"""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')
        
        # Prepare data
        data = [
            ['Metric', 'TGAT', 'TGN', 'Winner'],
            ['Total Parameters', f"{metrics['tgat'].get('total_parameters', 0):,}", 
             f"{metrics['tgn'].get('total_parameters', 0):,}",
             'TGAT' if metrics['tgat'].get('total_parameters', 0) < metrics['tgn'].get('total_parameters', 0) else 'TGN'],
            ['Model Size (MB)', f"{metrics['tgat'].get('model_size_mb', 0):.4f}", 
             f"{metrics['tgn'].get('model_size_mb', 0):.4f}",
             'TGAT' if metrics['tgat'].get('model_size_mb', 0) < metrics['tgn'].get('model_size_mb', 0) else 'TGN'],
            ['Inference Time (ms)', f"{metrics['tgat'].get('inference_time', 0):.2f}", 
             f"{metrics['tgn'].get('inference_time', 0):.2f}",
             'TGAT' if metrics['tgat'].get('inference_time', 0) < metrics['tgn'].get('inference_time', 0) else 'TGN'],
            ['Memory Usage (MB)', f"{metrics['tgat'].get('memory_usage', 0):.2f}", 
             f"{metrics['tgn'].get('memory_usage', 0):.2f}",
             'TGAT' if metrics['tgat'].get('memory_usage', 0) < metrics['tgn'].get('memory_usage', 0) else 'TGN'],
        ]
        
        # Create table
        table = ax.table(cellText=data, cellLoc='center', loc='center',
                        colWidths=[0.3, 0.2, 0.2, 0.2])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style winner column
        for i in range(1, len(data)):
            table[(i, 3)].set_facecolor('#f1c40f')
            table[(i, 3)].set_text_props(weight='bold')
        
        # Alternate row colors
        for i in range(1, len(data)):
            for j in range(3):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
                else:
                    table[(i, j)].set_facecolor('#ffffff')
        
        plt.title('TGAT vs TGN - Summary Comparison', fontsize=14, fontweight='bold', pad=20)
        
        filename = os.path.join(output_path, 'summary_table.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
        
        return filename
    
    @staticmethod
    def create_all_visualizations(metrics: Dict, output_path: str = 'comparison_outputs'):
        """Generate all visualizations"""
        
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80 + "\n")
        
        files = []
        files.append(MetricsVisualizer.plot_model_comparison(metrics, output_path))
        files.append(MetricsVisualizer.plot_speedup_analysis(metrics, output_path))
        files.append(MetricsVisualizer.plot_efficiency_metrics(metrics, output_path))
        files.append(MetricsVisualizer.plot_summary_table(metrics, output_path))
        
        print("\n" + "="*80)
        print(f"✓ ALL VISUALIZATIONS SAVED TO: {output_path}/")
        print("="*80 + "\n")
        
        return files
