# Quick Start Guide: Temporal Graph Model Comparison

## 🚀 Getting Started

This project compares 6 temporal graph neural network models:
- **TGAT** (Temporal Graph Attention Network)
- **TGN** (Temporal Graph Network)  
- **DyRep** (Dynamic Representation Learning)
- **EvolveGCN** (Evolving Graph Convolutional Networks)
- **JODIE** (Joint Dynamic User-Item Embedding)
- **CTDG** (Continuous-Time Dynamic Graph)

---

## 📋 Prerequisites

```bash
# Install required packages
pip install torch numpy matplotlib seaborn
```

---

## 🎯 Running the Comparison

### Option 1: Full Comprehensive Comparison (Recommended)

Run all 6 models with complete metrics and visualizations:

```bash
python comprehensive_comparison.py
```

**Output:**
- 7 visualization PNG files in `comparison_outputs/`
- Console output with detailed metrics
- Comparison of all models across multiple dimensions

**Time:** ~5-10 minutes on CPU

---

### Option 2: Quick Demo (TGAT vs TGN only)

For a faster comparison of just TGAT and TGN:

```bash
python compare_with_graphs.py
```

**Output:**
- 4 visualization PNG files
- Faster execution (~2-3 minutes)

---

## 📊 Generated Visualizations

After running `comprehensive_comparison.py`, you'll find these files in `comparison_outputs/`:

1. **model_sizes.png**
   - Bar charts showing parameter counts and model sizes
   - Helps identify the most compact models

2. **inference_speed.png**
   - Inference time comparison across models
   - Lower is better for real-time applications

3. **memory_usage.png**
   - Memory consumption during inference
   - Critical for deployment planning

4. **training_performance.png**
   - Training time for 5 epochs
   - Shows which models train fastest

5. **metrics_table.png**
   - Comprehensive table with all metrics
   - Easy reference for all measurements

6. **radar_chart.png**
   - Multi-dimensional performance visualization
   - Shows strengths/weaknesses at a glance

7. **efficiency_analysis.png**
   - Speed vs size tradeoff scatter plot
   - Identifies best balanced models

---

## 📈 Understanding the Results

### Model Size
- **Smallest:** EvolveGCN (~26K parameters, 0.10 MB)
- **Largest:** CTDG (~99K parameters, 0.38 MB)

### Training Speed
- **Fastest:** JODIE (~11.4s per epoch)
- **Slowest:** CTDG (~50.5s per epoch)

### Best Use Cases

| Model | Best For |
|-------|----------|
| **TGAT** | General temporal graph learning |
| **TGN** | Dynamic graph representation |
| **DyRep** | Dynamic social networks |
| **EvolveGCN** | Edge/mobile deployment (smallest) |
| **JODIE** | Real-time recommendations (fastest) |
| **CTDG** | High-accuracy temporal modeling |

---

## 🔧 Customizing the Comparison

Edit `comprehensive_comparison.py` to modify:

```python
config = {
    'num_nodes': 500,        # Number of nodes in graph
    'num_edges': 2000,       # Number of edges
    'num_timestamps': 30,    # Temporal snapshots
    'in_channels': 32,       # Input feature dimension
    'hidden_channels': 64,   # Hidden layer size
    'out_channels': 16,      # Output dimension
    'num_layers': 2,         # Number of layers
    'num_heads': 4,          # Attention heads
    'dropout': 0.1           # Dropout rate
}
```

---

## 📁 Project Structure

```
tgat_project/
├── comprehensive_comparison.py    # Main comparison script
├── compare_with_graphs.py         # Quick TGAT vs TGN demo
├── tgat_model.py                  # TGAT implementation
├── tgn_model.py                   # TGN implementation
├── dyrep_model.py                 # DyRep implementation
├── evolvegcn_model.py             # EvolveGCN implementation
├── jodie_model.py                 # JODIE implementation
├── ctdg_model.py                  # CTDG implementation
├── comparison_framework.py        # Metrics and profiling tools
├── visualizer.py                  # Visualization utilities
├── comparison_outputs/            # Generated visualizations
│   ├── model_sizes.png
│   ├── inference_speed.png
│   ├── memory_usage.png
│   ├── training_performance.png
│   ├── metrics_table.png
│   ├── radar_chart.png
│   └── efficiency_analysis.png
└── COMPREHENSIVE_COMPARISON_REPORT.md  # Detailed report
```

---

## 🎓 Model Implementations

Each model is implemented as a standalone PyTorch module:

### Testing Individual Models

```python
from tgat_model import TGAT
import torch

# Create model
model = TGAT(
    in_channels=32,
    hidden_channels=64,
    out_channels=16,
    num_layers=2,
    num_heads=4
)

# Generate sample data
x = torch.randn(100, 32)  # Node features
edge_index = torch.randint(0, 100, (2, 500))  # Edges
timestamps = torch.randint(0, 30, (500,))  # Timestamps

# Forward pass
output = model(x, edge_index, timestamps)
print(f"Output shape: {output.shape}")  # [100, 16]
```

---

## 🐛 Troubleshooting

### Issue: "Module not found"
```bash
# Make sure you're in the project directory
cd tgat_project

# Install dependencies
pip install torch numpy matplotlib seaborn
```

### Issue: "CUDA out of memory"
The comparison runs on CPU by default. If you want GPU:
```python
# In comprehensive_comparison.py, line ~700
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### Issue: Slow execution
- Reduce `num_nodes` and `num_edges` in config
- Reduce number of epochs in training comparison
- Use `compare_with_graphs.py` for faster results

---

## 📊 Interpreting Metrics

### Parameters
- **Lower is better** for deployment
- Affects model size and memory

### Inference Time
- **Lower is better** for real-time applications
- Measured in milliseconds per forward pass

### Memory Usage
- **Lower is better** for resource-constrained devices
- Measured in MB during inference

### Training Time
- **Lower is better** for rapid experimentation
- Measured in seconds per epoch

---

## 🎯 Next Steps

1. **View Results:** Open PNG files in `comparison_outputs/`
2. **Read Report:** Check `COMPREHENSIVE_COMPARISON_REPORT.md`
3. **Customize:** Modify config and re-run
4. **Experiment:** Try different datasets and configurations
5. **Deploy:** Choose the best model for your use case

---

## 📚 Additional Resources

- **TGAT Paper:** [Inductive Representation Learning on Temporal Graphs](https://arxiv.org/abs/2002.07962)
- **TGN Paper:** [Temporal Graph Networks](https://arxiv.org/abs/2006.10637)
- **DyRep Paper:** [Learning Representations over Dynamic Graphs](https://openreview.net/forum?id=HyePrhR5KX)
- **EvolveGCN Paper:** [EvolveGCN](https://arxiv.org/abs/1902.10191)
- **JODIE Paper:** [Predicting Dynamic Embedding Trajectory](https://arxiv.org/abs/1908.01207)

---

## 💡 Tips

- **Start with the quick demo** to understand the workflow
- **Use the radar chart** to quickly identify best models
- **Check the efficiency analysis** for speed/size tradeoffs
- **Read the comprehensive report** for detailed insights

---

## 🤝 Contributing

To add a new model:
1. Create `new_model.py` with PyTorch implementation
2. Add to `model_configs` in `comprehensive_comparison.py`
3. Ensure it follows the same interface (in_channels, hidden_channels, etc.)
4. Run comparison to generate updated visualizations

---

**Happy Comparing! 🚀**
