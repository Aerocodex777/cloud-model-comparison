
import torch
import traceback
from data_loader import load_autoscaling_data
from dyrep_model import DyRep
from evolvegcn_model import EvolveGCN

def debug_real_data():
    try:
        data = load_autoscaling_data('autoscaling_dataset.csv')
        x = data['x']
        edge_index = data['edge_index']
        timestamps = data['timestamps']
        # features?
        edge_attr = data['edge_attr']
        
        in_dim = x.shape[1]
        hidden = 64
        out_dim = len(torch.unique(data['y']))
        
        print("Data Shapes:", x.shape, edge_index.shape)

        with open("debug_error.log", "w") as f:
            f.write("Starting Debug\n")

            # DyRep
            try:
                print("Testing DyRep...")
                model = DyRep(in_dim, hidden, out_dim)
                # DyRep forward (x, edge_index, timestamps)
                out = model(x, edge_index, timestamps)
                f.write("DyRep Success\n")
            except Exception as e:
                f.write(f"DyRep Failed: {str(e)}\n")
                f.write(traceback.format_exc() + "\n")
            
            # EvolveGCN
            try:
                print("Testing EvolveGCN...")
                model = EvolveGCN(in_dim, hidden, out_dim)
                out = model(x, edge_index, timestamps)
                f.write("EvolveGCN Success\n")
            except Exception as e:
                f.write(f"EvolveGCN Failed: {str(e)}\n")
                f.write(traceback.format_exc() + "\n")

    except Exception as e:
        print("Data Load Failed:", e)

if __name__ == "__main__":
    debug_real_data()
