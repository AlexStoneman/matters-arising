from evaluation import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("run_name", type=str, help="Beginning of summary file name")
parser.add_argument("num_iterations", type=int, help="Number of iterations of training")
args = parser.parse_args()

run_name = args.run_name
num_iterations = args.num_iterations

for model_name in ["XGB","RF","MLP"]:
    for features_name in ["OH","FP","AGILEdesc"]:
        for cell_line in ["HeLa","RAW"]:
            for split_name in ["murcko","library"]:
                folder_name = os.path.join(run_name,model_name,features_name,cell_line,split_name)
                print("\n"+folder_name + " Results")
                if split_name=="murcko":
                    baseline_accuracy(folder_name,10,True,5)
                elif split_name=="library":
                    baseline_accuracy(folder_name,10,False,0)

