from models import *
import argparse
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

parser = argparse.ArgumentParser()
parser.add_argument("run_name", type=str, help="Beginning of summary file name")
parser.add_argument("num_iterations", type=int, help="Number of iterations of training")
parser.add_argument("gpu", type=str, help="Name of GPU used for training")
args = parser.parse_args()

run_name = args.run_name
num_iterations = args.num_iterations
gpu = args.gpu

for model_name in ["XGB","RF","MLP"]:
    for features_name in ["OH","FP","AGILEdesc"]:
        for cell_line in ["HeLa"]:
            for split_name in ["murcko","library"]:
                folder_name = os.path.join(run_name,model_name,features_name,cell_line,split_name)
                train_test_ensemble(folder_name,model_name,features_name,cell_line,split_name,num_iterations,gpu)
    
