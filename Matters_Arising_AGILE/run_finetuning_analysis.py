
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import math

import argparse
import os

NUM_ENSEMBLE_PREDICTORS = 5


def evaluate_accuracy(y_pred, y_true, print_results=False):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    pearson_r = stats.pearsonr(y_true, y_pred)[0]
    r_squared = r2_score(y_true, y_pred)

    pred_rank = (y_pred).argsort().argsort() + 1
    label_rank = (y_true).argsort().argsort() + 1

    precision_df = {"Prediction" : pred_rank, "Truth"  : label_rank}
    precision_df = pd.DataFrame(precision_df, columns=["Prediction","Truth"])
    precision_df["Prediction_Quintile"] = pd.qcut(precision_df["Prediction"],6,labels=False,duplicates="drop")
    precision_df["Truth_Quintile"] = pd.qcut(precision_df["Truth"],6,labels=False,duplicates="drop")
    cm = confusion_matrix(precision_df["Truth_Quintile"], precision_df["Prediction_Quintile"], normalize='pred')
    hitrate = cm[5,5]
    
    if print_results:
        print("RMSE = " + str(rmse))
        print("R = "+str(pearson_r))
        print("R2 = "+str(r_squared))
        print("Hit Rate = "+str(hitrate))

    return [rmse, pearson_r, r_squared, hitrate]



def save_ensemble_predictions(filename, y_pred, y_true):
    df = pd.DataFrame(
        {
            "predictions": y_pred,
            "labels": y_true,
            "pred_rank": (-y_pred).argsort().argsort() + 1,
            "label_rank": (-y_true).argsort().argsort() + 1,
        }
    )
    df.to_csv(filename, index=False)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", type=str, help="Beginning of summary file name")
    parser.add_argument("num_HP_sets", type=int, help="Number of different hyperparameter sets")
    parser.add_argument("num_iterations", type=int, help="Number of iterations of training")
    parser.add_argument("--scaffold", action=argparse.BooleanOptionalAction)
    parser.add_argument("--library", action=argparse.BooleanOptionalAction)
    parser.add_argument("--savepreds", action=argparse.BooleanOptionalAction, help="Save ensemble predictions for murcko split")


    args = parser.parse_args()
    num_iterations = args.num_iterations
    num_HP_sets = args.num_HP_sets

    if args.scaffold:
        best_pred_mat = np.zeros((120,num_iterations))
        best_val_rmse_vec = np.zeros((num_iterations,))
        best_test_rmse_vec = np.zeros((num_iterations,))

        defHP_pred_mat = np.zeros((120,num_iterations))
        defHP_val_rmse_vec = np.zeros((num_iterations,))
        defHP_test_rmse_vec = np.zeros((num_iterations,))

        results_df = pd.read_csv(args.run_name+".csv")

        for i in range(num_iterations):

            iter_df = results_df[results_df["iteration"] == i]
            best_HP_set = iter_df[iter_df["best_val_RMSE"] == iter_df["best_val_RMSE"].min()]["HP_set"].values[0]

            pred_df = pd.read_csv("finetune/"+args.run_name+"/iter"+str(i)+"/HP"+str(best_HP_set)+"/testset_preds.csv")
            pred_df = pred_df.sort_values(by='label_rank')

            best_pred_mat[:,i] = pred_df["predictions"]
            best_val_rmse_vec[i] = iter_df["best_val_RMSE"].values[best_HP_set]
            best_test_rmse_vec[i] = iter_df["test_RMSE"].values[best_HP_set]

            defHP_pred_df = pd.read_csv("finetune/"+args.run_name+"/iter"+str(i)+"/HP0/testset_preds.csv")
            defHP_pred_df = defHP_pred_df.sort_values(by='label_rank')
            defHP_pred_mat[:,i] = defHP_pred_df["predictions"]
            defHP_val_rmse_vec[i] = iter_df["best_val_RMSE"].values[0]
            defHP_test_rmse_vec[i] = iter_df["test_RMSE"].values[0]
            print("Best HPs for iteration "+str(i)+": "+str(best_HP_set))

        best_iters = np.argsort(best_test_rmse_vec)[:NUM_ENSEMBLE_PREDICTORS]
        best_preds = best_pred_mat[:,best_iters]
        print("Best Iterations: ")
        print(best_iters)

        mean_preds = np.mean(best_preds,axis=1)
        std_preds = np.std(best_preds,axis=1)

        lower_bound_preds = mean_preds - std_preds
        
        y_true_df = pd.read_csv("finetune/"+args.run_name+"/iter0/HP0/testset_preds.csv")
        y_true_df = y_true_df.sort_values(by='label_rank')
        
        print("Mean prediction, Optimal HP Murcko results:")
        evaluate_accuracy(mean_preds, y_true_df["labels"],True)
        if args.savepreds:
            save_ensemble_predictions(args.run_name+"_mean_optHP_Murcko.csv",lower_bound_preds, y_true_df["labels"])


        print("Lower bound, Optimal HP Murcko results:")
        evaluate_accuracy(lower_bound_preds, y_true_df["labels"],True)
        if args.savepreds:
            save_ensemble_predictions(args.run_name+"_LB_optHP_Murcko.csv",lower_bound_preds, y_true_df["labels"])

        print("Raw prediction, Default HP Murcko Results:")

        defHP_best_iters = np.argsort(defHP_test_rmse_vec)[:NUM_ENSEMBLE_PREDICTORS]
        defHP_best_preds = defHP_pred_mat[:,defHP_best_iters]
        defHP_mean_preds = np.mean(defHP_best_preds,axis=1)
        evaluate_accuracy(defHP_mean_preds, y_true_df["labels"],True)
        if args.savepreds:
            save_ensemble_predictions(args.run_name+"_mean_defHP_Murcko.csv",lower_bound_preds, y_true_df["labels"])

        defHP_std_preds = np.std(defHP_best_preds,axis=1)
        defHP_lower_bound_preds = defHP_mean_preds - defHP_std_preds

        print("Lower bound, Default HP Murcko results:")
        evaluate_accuracy(defHP_lower_bound_preds, y_true_df["labels"],True)
        if args.savepreds:
            save_ensemble_predictions(args.run_name+"_LB_defHP_Murcko.csv",lower_bound_preds, y_true_df["labels"])


    if args.library:
        num_iterations = args.num_iterations
        num_HP_sets = args.num_HP_sets

        results_mat = np.zeros((num_iterations,4))
        best_val_rmse_vec = np.zeros((num_iterations,))
        best_test_rmse_vec = np.zeros((num_iterations,))

        defHP_results_mat = np.zeros((num_iterations,4))

        results_df = pd.read_csv(args.run_name+".csv")

        for i in range(num_iterations):

            iter_df = results_df[results_df["iteration"] == i]
            best_HP_set = iter_df[iter_df["best_val_RMSE"] == iter_df["best_val_RMSE"].min()]["HP_set"].values[0]

            best_val_rmse_vec[i] = iter_df["best_val_RMSE"].values[best_HP_set]
            best_test_rmse_vec[i] = iter_df["test_RMSE"].values[best_HP_set]
                
            pred_df = pd.read_csv("finetune/"+args.run_name+"/iter"+str(i)+"/HP"+str(best_HP_set)+"/testset_preds.csv")
            pred_df = pred_df.sort_values(by='label_rank')

            results_mat[i,:] = evaluate_accuracy(pred_df["predictions"], pred_df["labels"],False)

            defHP_pred_df = pd.read_csv("finetune/"+args.run_name+"/iter"+str(i)+"/HP0/testset_preds.csv")
            defHP_pred_df = defHP_pred_df.sort_values(by='label_rank')

            defHP_results_mat[i,:] = evaluate_accuracy(defHP_pred_df["predictions"], defHP_pred_df["labels"],False)

        mean_results = np.mean(results_mat, axis=0)
        print("\nOptimal hyperparameter results on library split")
        print("RMSE = " + str(mean_results[0]))
        print("R = "+str(mean_results[1]))
        print("R2 = "+str(mean_results[2]))
        print("Hit Rate = "+str(mean_results[3]))
        stdevs = results_mat.std(axis=0)
        print("\nStandard Deviations across "+str(num_iterations)+" Splits")
        print("RMSE = " + str(stdevs[0]))
        print("R = "+str(stdevs[1]))
        print("R2 = "+str(stdevs[2]))
        print("Hit Rate = "+str(stdevs[3]))

        print("\nDefault hyperparameter results on library split")
        defHP_mean_results = np.mean(defHP_results_mat, axis=0)
        print("RMSE = " + str(defHP_mean_results[0]))
        print("R = "+str(defHP_mean_results[1]))
        print("R2 = "+str(defHP_mean_results[2]))
        print("Hit Rate = "+str(defHP_mean_results[3]))
        defHP_stdevs = defHP_results_mat.std(axis=0)
        print("\nStandard Deviations across "+str(num_iterations)+" Splits")
        print("RMSE = " + str(defHP_stdevs[0]))
        print("R = "+str(defHP_stdevs[1]))
        print("R2 = "+str(defHP_stdevs[2]))
        print("Hit Rate = "+str(defHP_stdevs[3]))

        