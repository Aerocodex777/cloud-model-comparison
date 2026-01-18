
import torch
import torch.nn as nn
from dyrep_model import DyRep
from evolvegcn_model import EvolveGCN

def debug_synthetic():
    num_nodes = 10
    num_edges = 20
    in_dim = 3
    hidden = 16
    out_dim = 2
    
    x = torch.randn(num_nodes, in_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    timestamps = torch.arange(num_edges).float()
    
    print("--- Debugging DyRep ---")
    try:
        model = DyRep(in_dim, hidden, out_dim)
        out = model(x, edge_index, timestamps)
        print("DyRep Success! Shape:", out.shape)
    except Exception as e:
        print("DyRep Fail:", e)
        # trace
        import traceback
        traceback.print_exc()

    print("\n--- Debugging EvolveGCN ---")
    try:
        model = EvolveGCN(in_dim, hidden, out_dim)
        out = model(x, edge_index, timestamps)
        print("EvolveGCN Success! Shape:", out.shape)
    except Exception as e:
        print("EvolveGCN Fail:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_synthetic()
