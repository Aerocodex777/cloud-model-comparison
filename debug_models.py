
import torch
from data_loader import load_autoscaling_data
from dyrep_model import DyRep
from evolvegcn_model import EvolveGCN

def test_models():
    data = load_autoscaling_data('autoscaling_dataset.csv')
    x = data['x']
    edge_index = data['edge_index']
    timestamps = data['timestamps']
    num_nodes = data['num_nodes']
    in_dim = x.shape[1]
    hidden = 64
    out_dim = len(torch.unique(data['y']))

    print("Testing DyRep...")
    try:
        model = DyRep(in_dim, hidden, out_dim)
        out = model(x, edge_index, timestamps)
        print("DyRep Output shape:", out.shape)
    except Exception as e:
        print("DyRep Failed:", e)
        import traceback
        traceback.print_exc()

    print("\nTesting EvolveGCN...")
    try:
        model = EvolveGCN(in_dim, hidden, out_dim)
        out = model(x, edge_index, timestamps)
        print("EvolveGCN Output shape:", out.shape)
    except Exception as e:
        print("EvolveGCN Failed:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_models()
