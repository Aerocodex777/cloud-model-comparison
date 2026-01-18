# Comprehensive Temporal Graph Model Comparison

## Microservice Level Evaluation

| Model | Inference Latency (ms) | Training Time (s) | Memory (MB) | Accuracy |
|-------|------------------------|-------------------|-------------|----------|
| **TGN** | 52.01 | 0.25 | 138.58 | 1.0000 |
| **Transformer** | 4.38 | 0.01 | 2.48 | 1.0000 |
| **JODIE** | 2174.01 | 7.62 | 309.20 | 1.0000 |
| **DyRep** | FAILED | FAILED | FAILED | FAILED |
| **TGAT** | 10.75 | 0.03 | 19.07 | 1.0000 |
| **EvolveGCN** | 637.37 | 2.62 | 194.48 | 1.0000 |

## Metric Definitions
- **Temporal Modeling**: Ability to update state based on interaction history (e.g., JODIE/TGN).
- **Scalability**: Performance stability as data size grows (Transformer excels here).
- **Computational Efficiency**: Balance of high speed and low memory (EvolveGCN/Transformer).
- **Explainability**: Models like TGAT and Transformer offer attention weights for interpretation.
