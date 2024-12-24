import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from splits import *
import os


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


def evaluate_accuracy(y_pred, y_true, print_results):
    rmse = root_mean_squared_error(y_true, y_pred)
    pearson_r = stats.pearsonr(y_true, y_pred).statistic
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

    return  [rmse, pearson_r, r_squared, hitrate]



def baseline_accuracy(folder_name, num_iterations, ensemble, num_predictors):

    if ensemble:
        best_pred_mat = np.zeros((120,10))
        best_test_rmse_vec = np.zeros((10,))

        for i in range(num_iterations):

            pred_df = pd.read_csv(os.path.join(folder_name,"iter"+str(i),"testset_preds.csv"))
            pred_df = pred_df.sort_values(by='labels')

            best_pred_mat[:,i] = pred_df["predictions"]
            best_test_rmse_vec[i] = root_mean_squared_error(pred_df["predictions"],pred_df["labels"])

        best_iters = np.argsort(best_test_rmse_vec)[:num_predictors]
        best_preds = best_pred_mat[:,best_iters]
        mean_preds = np.mean(best_preds,axis=1)

        y_true_df = pd.read_csv(os.path.join(folder_name,"iter0","testset_preds.csv"))
        y_true_df = y_true_df.sort_values(by='labels')
        print("Model: "+folder_name)
        mean_results = evaluate_accuracy(mean_preds, y_true_df["labels"],True)
        save_ensemble_predictions(os.path.join(folder_name,"ensemble_preds.csv"),mean_preds,y_true_df["labels"]) 

    else:
        all_results = np.zeros((num_iterations,4))
        for i in range(num_iterations):
            y_pred_filename = os.path.join(folder_name,"iter"+str(i),"y_pred.csv")
            y_true_filename = os.path.join(folder_name,"iter"+str(i),"y_true.csv")
            y_pred = np.loadtxt(y_pred_filename, delimiter=',')
            y_true = np.loadtxt(y_true_filename, delimiter=',')
            all_results[i,:] = np.array((evaluate_accuracy(y_pred, y_true,False)))
        mean_results = all_results.mean(axis=0)
        print("Model: "+folder_name)
        print("Means across "+str(num_iterations)+" Splits")
        print("RMSE = " + str(mean_results[0]))
        print("R = "+str(mean_results[1]))
        print("R2 = "+str(mean_results[2]))
        print("Hit Rate = "+str(mean_results[3]))
        stdevs = all_results.std(axis=0)
        print("Standard Deviations across "+str(num_iterations)+" Splits")
        print("RMSE = " + str(stdevs[0]))
        print("R = "+str(stdevs[1]))
        print("R2 = "+str(stdevs[2]))
        print("Hit Rate = "+str(stdevs[3]))

