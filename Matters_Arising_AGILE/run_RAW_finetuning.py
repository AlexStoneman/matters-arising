
import yaml
from finetune import *
import csv
import argparse

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", type=str, help="Beginning of summary file name")
    parser.add_argument("num_iterations", type=int, help="Number of iterations of training")
    parser.add_argument("num_HP_sets", type=int, help="Number of different hyperparameter sets")
    parser.add_argument("gpu", type=str, help="Name of GPU used for training")
    args = parser.parse_args()

    run_name = args.run_name
    num_HP_sets = args.num_HP_sets
    num_iterations = args.num_iterations
    gpu = args.gpu
    

    for split in ["scaffold","library"]:
        for cell_line in ["RAW"]:

            summary_file = run_name+"_"+split+"_"+cell_line+".csv"

            if cell_line == "HeLa":
                task_name = "lnp_hela_with_feat"
                target_list = ["expt_Hela"]
            elif cell_line == "RAW":
                task_name = "lnp_raw_with_feat"
                target_list = ["expt_Raw"]

            with open(summary_file, 'w', newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["iteration","HP_set","best_val_RMSE","test_RMSE"])

            for i in range(num_iterations):

                for HP_set_idx in range(num_HP_sets):

                    config_file_name = "config_files/config_finetune_HP"+str(HP_set_idx)+".yaml"
                    config = yaml.load(open(config_file_name, "r"), Loader=yaml.FullLoader)

                    # default config additions 
                    config["task_name"] = task_name
                    config["dataset"]["target"] = target_list[0]

                    config["dataset"]["task"] = "regression"
                    config["dataset"]["data_path"] = "data/finetuning_set_smiles_plus_features.csv"
                    config["dataset"]["feature_cols"] = get_desc_cols(
                        config["dataset"]["data_path"]
                    )
                    config["model"]["pred_additional_feat_dim"] = len(
                        config["dataset"]["feature_cols"]
                        )

                    # new config additions for organization and reproducibility
                    config["seed"] = i
                    config["gpu"] = gpu
                    config["dir_name_root"] = split+"_iter"+str(i)+"_HP"+str(HP_set_idx)
                    config["summary_file"] = summary_file
                    config["dataset"]["splitting"] = split
                    config["dir_header"] = args.run_name + "_"+split + "_"+cell_line+"/iter" + str(i)+"/HP"+str(HP_set_idx)
                    if split == "library":
                        config["dataset"]["library_split_seed"] = i

                    print(config)

                    results_list = []

                    torch.manual_seed(seed = config["seed"])
                    torch.cuda.manual_seed(seed = config["seed"])
                    torch.cuda.manual_seed_all(seed = config["seed"])
                    np.random.seed(config["seed"])
                    random.seed(config["seed"])
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                    torch.use_deterministic_algorithms(True, warn_only=True)
                    config["dataset"]["generator_seed"] = config["seed"]

                    
                    for target in target_list:
                        config["dataset"]["target"] = target
                        finetune_agent = main(config)
                        test_RMSE = finetune_agent.res
                        valid_RMSE = finetune_agent.best_val_loss
                        # results_list.append([target, result])

                    with open(summary_file, 'a', newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([i,HP_set_idx,valid_RMSE,test_RMSE])
                                
                    print(f"Results saved to {finetune_agent.writer.log_dir}")