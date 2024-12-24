import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import argparse
from sklearn.metrics import confusion_matrix

import itertools
from mpl_toolkits.axes_grid1 import ImageGrid


def get_confusion_matrix(y_pred, y_true, num_sectors):
    precision_df = {"Prediction" : y_pred, "Truth"  : y_true}
    precision_df = pd.DataFrame(precision_df, columns=["Prediction","Truth"])
    precision_df["Prediction_Quintile"] = pd.qcut(precision_df["Prediction"],num_sectors,labels=False)
    precision_df["Truth_Quintile"] = pd.qcut(precision_df["Truth"],num_sectors,labels=False)
    cm = confusion_matrix(precision_df["Truth_Quintile"], precision_df["Prediction_Quintile"], normalize='pred')
    return cm


def get_AGILE_full_lib_df():

    full_pred_filenames = [
        "Matters_Arising_AGILE/finetune/AGILE_MA_scaffold_HeLa/iter0/HP1/preds_on_lnp_hela_with_feat.csv",
        "Matters_Arising_AGILE/finetune/AGILE_MA_scaffold_HeLa/iter3/HP0/preds_on_lnp_hela_with_feat.csv",
        "Matters_Arising_AGILE/finetune/AGILE_MA_scaffold_HeLa/iter4/HP1/preds_on_lnp_hela_with_feat.csv",
        "Matters_Arising_AGILE/finetune/AGILE_MA_scaffold_HeLa/iter1/HP1/preds_on_lnp_hela_with_feat.csv",
        "Matters_Arising_AGILE/finetune/AGILE_MA_scaffold_HeLa/iter9/HP1/preds_on_lnp_hela_with_feat.csv",
    ]
    all_pred_mat = np.zeros((1200,5))

    for i in range(len(full_pred_filenames)):
        filename = full_pred_filenames[i]
        full_pred_df = pd.read_csv(filename)
        full_pred_df = full_pred_df.sort_values(by='label_rank')
        all_pred_mat[:,i] = full_pred_df["predictions"]
    
    mean_preds = np.mean(all_pred_mat,axis=1)
    std_preds = np.std(all_pred_mat,axis=1)
    lower_bound_preds = mean_preds - std_preds
        
    y_true_df = pd.read_csv("Matters_Arising_AGILE/finetune/AGILE_MA_scaffold_HeLa/iter0/HP1/preds_on_lnp_hela_with_feat.csv")
    y_true_df = y_true_df.sort_values(by='label_rank')
    y_true = y_true_df["labels"]

    full_df = pd.DataFrame(
        {
            "predictions": lower_bound_preds,
            "labels": y_true,
            "pred_rank": (-lower_bound_preds).argsort().argsort() + 1,
            "label_rank": (-y_true).argsort().argsort() + 1,
        }
    )
    return full_df

def make_parity_plots(filename_root):

    AGILE_df_filename = "Matters_Arising_AGILE/AGILE_MA_scaffold_HeLa_LB_optHP_Murcko.csv"
    baseline_df_filename = "Matters_Arising_Baselines/MA_results/XGB/OH/HeLa/murcko/ensemble_preds.csv"

    AGILE_df = pd.read_csv(AGILE_df_filename)
    baseline_df = pd.read_csv(baseline_df_filename)

    AGILE_y_pred = AGILE_df["predictions"]
    AGILE_y_true = AGILE_df["labels"]    
    baseline_y_pred = baseline_df["predictions"]
    baseline_y_true = baseline_df["labels"]


    fig, (ax1,ax2) = plt.subplots(1,2, sharey=True)
    fig.set_figheight(2.9)
    fig.set_figwidth(5)

    ax1.scatter(AGILE_y_true, AGILE_y_pred, s=10)
    ax1.plot([-2,12],[-2,12],'r--')
    ax2.scatter(baseline_y_true,baseline_y_pred,s=10)
    ax2.plot([-2,12],[-2,12],'r--')
    ax2.set_xlabel("True Transfection Efficiency",fontsize = 9,font="Arial",weight="bold")
    ax1.set_xlabel("True Transfection Efficiency",fontsize = 9,font="Arial",weight="bold")
    ax1.set_ylabel("Predicted Transfection Efficiency",fontsize = 9,font="Arial",weight="bold")
    ax2.set_title("Best Baseline on Test Set",fontsize = 9,font="Arial",weight="bold")
    ax1.set_title("Best AGILE on Test Set",fontsize = 9,font="Arial",weight="bold")
    ax1.tick_params(axis='both', which='major', labelsize=9,labelfontfamily="Arial")
    ax2.tick_params(axis='both', which='major', labelsize=9,labelfontfamily="Arial")
    ax1.set_xticks([-2,0,2,4,6,8,10,12])
    ax2.set_xticks([-2,0,2,4,6,8,10,12])
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.savefig(filename_root+"_parity_plot.png", format="png", dpi=1200, bbox_inches='tight')


def make_precision_matrices(filename_root):

    AGILE_full_lib_df = get_AGILE_full_lib_df()
    AGILE_test_df_filename = "Matters_Arising_AGILE/AGILE_MA_scaffold_HeLa_LB_optHP_Murcko.csv"
    baseline_df_filename = "Matters_Arising_Baselines/MA_results/XGB/OH/HeLa/murcko/ensemble_preds.csv"

    AGILE_test_df = pd.read_csv(AGILE_test_df_filename)
    baseline_df = pd.read_csv(baseline_df_filename)

    AGILE_full_y_pred = AGILE_full_lib_df["pred_rank"]
    AGILE_full_y_true = AGILE_full_lib_df["label_rank"]
    AGILE_test_y_pred = AGILE_test_df["pred_rank"]
    AGILE_test_y_true = AGILE_test_df["label_rank"]    
    baseline_y_pred = baseline_df["pred_rank"]
    baseline_y_true = baseline_df["label_rank"]

    AGILE_all_cm = get_confusion_matrix(AGILE_full_y_pred, AGILE_full_y_true, 6)
    AGILE_test_cm = get_confusion_matrix(AGILE_test_y_pred, AGILE_test_y_true, 6)
    baseline_cm = get_confusion_matrix(baseline_y_pred, baseline_y_true, 6)

    y_classes = ["0 - 17%","17 - 33%","33 - 50%","50 - 67%","67 - 83%","83 - 100%"]
    y_classes.reverse()
    x_classes = ["0 - 17%","17 - 33%","33 - 50%","50 - 67%","67 - 83%","83 - 100%"]

    fig = plt.figure(figsize=(15,20))
    grid = ImageGrid(fig, 111, 
                    nrows_ncols=(1,3),
                    axes_pad=0.15,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="7%",
                    cbar_pad=0.15,
                    )

    cm_list = [np.rot90(AGILE_all_cm,3), np.rot90(AGILE_test_cm,3), np.rot90(baseline_cm,3)]
    title_list = ["Best AGILE on Full 1,200","Best AGILE on Test Set", "Best Baseline on Test Set"]

    title_fontdict = {'family':'Arial', 'weight':'bold','size':20}

    for i, ax in enumerate(grid[:3]):
        cm = cm_list[i]
        im = ax.imshow(cm, vmin=0, vmax=1)
        ax.set_title(title_list[i],font="Arial",fontsize=16,weight="bold")  
        tick_marks = np.arange(6)
        ax.set_xticks(tick_marks)  
        ax.set_xticklabels(x_classes, rotation=45,fontsize=14,font="Arial")
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(y_classes,fontsize=14,font="Arial")

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i+0.1, format(cm[i, j], '.2f'),
                    horizontalalignment="center",
                    color="white",
                    fontsize=14,
                    font="Arial",
                )

        ax.set_xlabel('True Percentile',fontsize=16,font="Arial",weight="bold")
        ax.set_ylabel('Predicted Percentile',fontsize=16,font="Arial",weight="bold")

    fig.subplots_adjust(right=0.8)
    cbar = fig.colorbar(im, cax=ax.cax)
    cbar.ax.tick_params(labelfontfamily="Arial",labelsize=14)

    plt.savefig(filename_root+"_precision_mats.png", format="png", dpi=1200, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("outfile_name",type=str)
    args = parser.parse_args()

    make_parity_plots(args.outfile_name)
    make_precision_matrices(args.outfile_name)