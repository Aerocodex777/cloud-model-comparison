
import torch
import torch.nn as nn
import torch.optim as optim
import time
import psutil
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

from data_loader import load_autoscaling_data
from tgn_model import TGN
from transformer_model import TemporalGraphTransformer
from jodie_model import JODIE
from dyrep_model import DyRep
from tgat_model import TGAT
from evolvegcn_model import EvolveGCN

# Set seaborn style for premium aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [14, 10]
plt.rcParams['font.size'] = 12

class ComprehensiveComparator:
    def __init__(self, data_path='autoscaling_dataset.csv', device='cpu'):
        self.device = device
        self.data = load_autoscaling_data(data_path)
        self.results = {}
        
        # Move data to device
        for k, v in self.data.items():
            if torch.is_tensor(v):
                self.data[k] = v.to(device)
    
    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB

    def train_evaluate(self, model_name, model, epochs=5):
        print(f"\n--- Training {model_name} ---")
        try:
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            num_edges = self.data['edge_index'].size(1)
            train_idx = int(num_edges * 0.7)
            val_idx = int(num_edges * 0.85)
            
            metrics = {
                'train_loss': [],
                'val_acc': [],
                'epoch_times': [],
                'inference_latency': [],
                'memory_usage_mb': 0
            }
            
            start_mem = self.get_memory_usage()
            
            model.train()
            for epoch in range(epochs):
                start_time = time.time()
                optimizer.zero_grad()
                
                # Model agnostic forward call
                # Some models take edge_attr, some don't.
                # All take x, edge_index, timestamps
                try:
                    # Try passing edge_attr if the model accepts it
                    out = model(self.data['x'], self.data['edge_index'], self.data['timestamps'], edge_attr=self.data['edge_attr'])
                except TypeError:
                    # Fallback for models not supporting edge_attr in forward
                    out = model(self.data['x'], self.data['edge_index'], self.data['timestamps'])
                
                loss = criterion(out[self.data['edge_index'][0, :train_idx]], self.data['y'][:train_idx])
                loss.backward()
                optimizer.step()
                
                epoch_time = time.time() - start_time
                metrics['train_loss'].append(loss.item())
                metrics['epoch_times'].append(epoch_time)
                
                # Validation
                model.eval()
                with torch.no_grad():
                    pred = out.argmax(dim=-1)
                    val_acc = accuracy_score(self.data['y'][train_idx:val_idx].cpu(), pred[self.data['edge_index'][0, train_idx:val_idx]].cpu())
                    metrics['val_acc'].append(val_acc)
                model.train()
                print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s | Acc: {val_acc:.4f}")
            
            metrics['memory_usage_mb'] = max(0, self.get_memory_usage() - start_mem)
            
            # Inference Latency
            model.eval()
            latencies = []
            with torch.no_grad():
                for _ in range(50):
                    t0 = time.time()
                    try:
                        _ = model(self.data['x'], self.data['edge_index'], self.data['timestamps'], edge_attr=self.data['edge_attr'])
                    except TypeError:
                        _ = model(self.data['x'], self.data['edge_index'], self.data['timestamps'])
                    latencies.append((time.time() - t0) * 1000)
            metrics['inference_latency'] = np.mean(latencies)
            
            self.results[model_name] = metrics
            
        except Exception as e:
            print(f"FAILED {model_name}: {e}")
            import traceback
            traceback.print_exc()
            with open("model_failures.log", "a") as errf:
                errf.write(f"--- {model_name} Failed ---\n")
                errf.write(str(e) + "\n")
                errf.write(traceback.format_exc() + "\n")
            
            # Record failure in results for report
            self.results[model_name] = {
                'inference_latency': -1,
                'epoch_times': [-1],
                'memory_usage_mb': -1,
                'val_acc': [-1]
            }

    def run_all(self):
        # Config
        in_dim_node = self.data['node_feature_dim']
        hidden_dim = 64
        out_dim = len(torch.unique(self.data['y']))
        
        # Instantiate Models
        models = [
            ("TGN", TGN(in_channels=in_dim_node, hidden_channels=hidden_dim, out_channels=out_dim, time_dim=16).to(self.device)),
            ("Transformer", TemporalGraphTransformer(in_channels=in_dim_node, hidden_channels=hidden_dim, out_channels=out_dim).to(self.device)),
            ("JODIE", JODIE(in_channels=in_dim_node, hidden_channels=hidden_dim, out_channels=out_dim).to(self.device)),
            ("DyRep", DyRep(in_channels=in_dim_node, hidden_channels=hidden_dim, out_channels=out_dim).to(self.device)),
            ("TGAT", TGAT(in_channels=in_dim_node, hidden_channels=hidden_dim, out_channels=out_dim).to(self.device)),
            ("EvolveGCN", EvolveGCN(in_channels=in_dim_node, hidden_channels=hidden_dim, out_channels=out_dim).to(self.device))
        ]
        
        # Clear log
        open("model_failures.log", "w").close()
        
        for name, model in models:
            self.train_evaluate(name, model)

        self.generate_report()
        self.generate_visualizations()

    def generate_report(self):
        with open("FULL_MODEL_COMPARISON.md", "w") as f:
            f.write("# Comprehensive Temporal Graph Model Comparison\n\n")
            f.write("## Microservice Level Evaluation\n\n")
            f.write("| Model | Inference Latency (ms) | Training Time (s) | Memory (MB) | Accuracy |\n")
            f.write("|-------|------------------------|-------------------|-------------|----------|\n")
            
            # Order to match execution or sorted
            ordered_names = ["TGN", "Transformer", "JODIE", "DyRep", "TGAT", "EvolveGCN"]
            for name in ordered_names:
                res = self.results.get(name)
                if not res or res['inference_latency'] == -1:
                    f.write(f"| **{name}** | FAILED | FAILED | FAILED | FAILED |\n")
                else:
                    start_mem = res.get('memory_usage_mb', 0)
                    lat = res.get('inference_latency', 0)
                    trn = np.mean(res.get('epoch_times', [0]))
                    acc = max(res.get('val_acc', [0]))
                    f.write(f"| **{name}** | {lat:.2f} | {trn:.2f} | {start_mem:.2f} | {acc:.4f} |\n")
                
            f.write("\n## Metric Definitions\n")
            f.write("- **Temporal Modeling**: Ability to update state based on interaction history (e.g., JODIE/TGN).\n")
            f.write("- **Scalability**: Performance stability as data size grows (Transformer excels here).\n")
            f.write("- **Computational Efficiency**: Balance of high speed and low memory (EvolveGCN/Transformer).\n")
            f.write("- **Explainability**: Models like TGAT and Transformer offer attention weights for interpretation.\n")

    def generate_visualizations(self):
        # 1. Bar Chart: Inference Latency
        plt.figure()
        names = list(self.results.keys())
        latencies = [self.results[n]['inference_latency'] for n in names]
        sns.barplot(x=names, y=latencies, palette="viridis")
        plt.title('Inference Latency (Lower is Better)')
        plt.ylabel('Latency (ms)')
        plt.savefig('full_latency_comparison.png')
        plt.close()
        
        # 2. Bubble Chart: Memory vs Latency
        plt.figure()
        memories = [self.results[n]['memory_usage_mb'] for n in names]
        plt.scatter(latencies, memories, s=1000, alpha=0.6, c=range(len(names)), cmap="tab10")
        for i, txt in enumerate(names):
            plt.annotate(txt, (latencies[i], memories[i]), ha='center', va='center')
        plt.title('Efficiency Landscape: Memory vs Latency')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Memory Usage (MB)')
        plt.grid(True)
        plt.savefig('full_efficiency_bubble.png')
        plt.close()
        
        # 3. Training & Accuracy
        plt.figure()
        for name in names:
            plt.plot(self.results[name]['val_acc'], marker='o', label=name)
        plt.title('Validation Accuracy over Epochs')
        plt.legend()
        plt.savefig('full_accuracy_plot.png')
        plt.close()

if __name__ == "__main__":
    comp = ComprehensiveComparator()
    comp.run_all()
