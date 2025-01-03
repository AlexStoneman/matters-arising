# Copyright (c) 2023 Shihao Ma, Haotian Cui, WangLab @ U of T

# inference and visualization
from io import StringIO
import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr
from rdkit import Chem
import rdkit.Chem.Draw as Draw
import matplotlib
from matplotlib import pyplot as plt
# import umap

from dataset.dataset_test import MolTestDatasetWrapper, MolTestDataset
from utils.plot import facecolors_customize
from utils.constants import (
    R2_to_type,
    R3_to_type,
    R2_to_chain_length,
    R3_to_chain_length,
)


apex_support = False
try:
    sys.path.append("./apex")
    from apex import amp

    apex_support = True
except:
    print(
        "Please install apex for mixed precision training from: https://github.com/NVIDIA/apex"
    )
    apex_support = False


class Normalizer(object):
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


def get_desc_cols(fname):
    """Get the descriptor columns from a csv file."""
    df = pd.read_csv(fname)
    return [
        col
        for col in df.columns
        if col not in ["smiles", "expt_Hela", "expt_Raw", "label", "labels"]
    ]


class Inference(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        self.log_dir = os.path.join("finetune", config["model_to_evaluate"])
        self.dataset = dataset
        self.criterion = nn.MSELoss()

    def _get_device(self):
        if torch.cuda.is_available() and self.config["gpu"] != "cpu":
            device = self.config["gpu"]
            torch.cuda.set_device(device)
        else:
            device = "cpu"
        print("Running on:", device)

        return device

    def _step(self, model, data, n_iter):
        # get the prediction
        __, pred = model(data)  # [N,C]

        if self.config["dataset"]["task"] == "classification":
            loss = self.criterion(pred, data.y.flatten())
        elif self.config["dataset"]["task"] == "regression":
            if self.normalizer:
                loss = self.criterion(pred, self.normalizer.norm(data.y))
            else:
                loss = self.criterion(pred, data.y)

        return loss

    def inference(self):
        data_loader = self.dataset.get_fulldata_loader()

        self.normalizer = None

        from models.agile_finetune import AGILE
        model = AGILE(self.config["dataset"]["task"], **self.config["model"]).to(
            self.device
        )
        model = self._load_pre_trained_weights(model)

        self.model = model

        layer_list = []
        for name, param in model.named_parameters():
            if "pred_" in name:
                print(name, param.requires_grad)
                layer_list.append(name)

        params = list(
            map(
                lambda x: x[1],
                list(filter(lambda kv: kv[0] in layer_list, model.named_parameters())),
            )
        )
        base_params = list(
            map(
                lambda x: x[1],
                list(
                    filter(lambda kv: kv[0] not in layer_list, model.named_parameters())
                ),
            )
        )

        # # save config file
        # _save_config_file(model_checkpoints_folder)

        predictions = []
        embeddings = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(data_loader):
                data = data.to(self.device)

                emb, pred = model(data)
                loss = self._step(model, data, bn)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config["dataset"]["task"] == "classification":
                    pred = F.softmax(pred, dim=-1)

                if self.device == "cpu":
                    predictions.extend(pred.detach().numpy())
                    embeddings.extend(emb.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    embeddings.extend(emb.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            test_loss /= num_data

        model.train()

        if self.config["dataset"]["task"] == "regression":
            predictions = np.array(predictions).flatten()
            embeddings = np.array(embeddings)
            labels = np.array(labels)
            if self.config["task_name"] in ["qm7", "qm8", "qm9"]:
                self.mae = mean_absolute_error(labels, predictions)
            else:
                self.rmse = mean_squared_error(labels, predictions, squared=False)
            self.corr = pearsonr(labels, predictions)[0]
            print(
                "Test loss:",
                test_loss,
                "Test RMSE:",
                self.rmse,
                "Test Corr:",
                self.corr,
            )

        elif self.config["dataset"]["task"] == "classification":
            predictions = np.array(predictions)
            embeddings = np.array(embeddings)
            labels = np.array(labels)
            self.roc_auc = roc_auc_score(labels, predictions[:, 1])
            print("Test loss:", test_loss, "Test ROC AUC:", self.roc_auc)

        # save predictions and labels to csv
        all_smiles = self.dataset.all_smiles
        pred_to_save = predictions if predictions.ndim == 1 else predictions[:, 1]
        assert len(all_smiles) == len(pred_to_save)
        df = pd.DataFrame(
            {
                "smiles": all_smiles,
                "predictions": pred_to_save,
                "labels": labels,
                "pred_rank": (-pred_to_save).argsort().argsort() + 1,
                "label_rank": (-labels).argsort().argsort() + 1,
            }
        )
        df.to_csv(
            os.path.join(self.log_dir, f"preds_on_{config['task_name']}.csv"),
            index=False,
        )

        return predictions, embeddings, labels

    def _eval_stratified_classes(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        q=6,
        use_set="all",
    ):
        """
        Evaluate the stratified classes of the predictions. If labels and predictions
        are intergers, will directly view them as categories. If labels and predictions
        are floats, will first convert them to categories by stratifying them.

        The stratification is done by sorting the predictions and labels in descending
        order and then split them into 5 groups with equal number of samples based
        on the quantiles.

        Args:
            labels (np.ndarray): Labels of the samples
            predictions (np.ndarray): Predictions of the samples
            q (int): Number of quantiles to split the samples into. Defaults to 4.
            use_set (str): The set to evaluate on, choices from train, test, all.
                Defaults to "test".
        """

        # fmt: off
        test_set_index = [45, 46, 47, 48, 49, 55, 56, 57, 58, 59, 145, 146, 147, 148, 149, 155, 156, 157, 
            158, 159, 245, 246, 247, 248, 249, 255, 256, 257, 258, 259, 345, 346, 347, 348, 349, 355, 356, 
            357, 358, 359, 445, 446, 447, 448, 449, 455, 456, 457, 458, 459, 545, 546, 547, 548, 549, 555, 
            556, 557, 558, 559, 645, 646, 647, 648, 649, 655, 656, 657, 658, 659, 745, 746, 747, 748, 749, 
            755, 756, 757, 758, 759, 845, 846, 847, 848, 849, 855, 856, 857, 858, 859, 945, 946, 947, 948, 
            949, 955, 956, 957, 958, 959, 1045, 1046, 1047, 1048, 1049, 1055, 1056, 1057, 1058, 1059, 1145, 
            1146, 1147, 1148, 1149, 1155, 1156, 1157, 1158, 1159]
        valid_set_index = [65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 165, 166, 167, 168, 169, 170, 171, 172, 
            173, 174, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 365, 366, 367, 368, 369, 370, 371, 
            372, 373, 374, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 565, 566, 567, 568, 569, 570, 
            571, 572, 573, 574, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 765, 766, 767, 768, 769, 
            770, 771, 772, 773, 774, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 965, 966, 967, 968, 
            969, 970, 971, 972, 973, 974, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1165, 
            1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174]
        # fmt: on
        if use_set == "test":
            labels = labels[test_set_index]
            predictions = predictions[test_set_index]
        elif use_set == "valid":
            labels = labels[valid_set_index]
            predictions = predictions[valid_set_index]
        elif use_set == "valid_test":
            labels = labels[valid_set_index + test_set_index]
            predictions = predictions[valid_set_index + test_set_index]
        elif use_set == "train":
            labels = np.delete(labels, test_set_index + valid_set_index)
            predictions = np.delete(predictions, test_set_index + valid_set_index)

        # if labels and predictions are floats, convert them to categories
        if not isinstance(labels[0], int) and not isinstance(predictions[0], int):
            labels = pd.qcut(labels, q, labels=False, duplicates="drop")
            predictions = pd.qcut(predictions, q, labels=False, duplicates="drop")

        # calculate the number of samples in each category
        num_samples = len(labels)
        num_samples_in_each_category = [
            len(labels[labels == i]) for i in np.unique(labels)
        ]
        assert len(labels) == len(predictions)
        print(
            "Number of samples in each category:",
            num_samples_in_each_category,
            "Total number of samples:",
            num_samples,
        )

        # compute accuracy, precision, recall, macro F1 score
        acc = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average="macro")
        recall = recall_score(labels, predictions, average="macro")
        f1_macro = f1_score(labels, predictions, average="macro")
        f1_micro = f1_score(labels, predictions, average="micro")
        print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}, F1_macro: {f1_macro:.4f}, F1_micro: {f1_micro:.4f}")
        self.stratified_class_results = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "macro f1": f1_macro,
            "micro f1": f1_micro,
        }

        import csv
        # specify the file name
        filename = os.path.join(
                self.log_dir,
                f"matrics_{use_set}_{self.config['task_name']}.csv",
            )

        # writing to csv file
        with open(filename, 'w') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            # writing the headers
            csvwriter.writerow(self.stratified_class_results.keys())

            # writing the data rows
            csvwriter.writerow(self.stratified_class_results.values())


        # compute and plot confusion matrix using seaborn api
        labels_to_show = np.sort(np.unique(labels))
        num_cates = len(labels_to_show)
        cm = confusion_matrix(labels, predictions, labels=labels_to_show)
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # import seaborn as sns

        # sns.set_style("white")
        # sns.set_context("paper", font_scale=1.5)
        # fig, ax = plt.subplots(figsize=(10, 8))
        # sns.heatmap(
        #     cm,
        #     # cm[::-1, :],  # reverse the y axis for better visualization
        #     annot=True,
        #     cmap="Blues",
        #     fmt=".2f",
        # )
        # # ax.set_title("Confusion Matrix")
        # ax.set_xlabel("Predicted top k percentiles")
        # ax.set_ylabel("Actual top k percentiles")
        # # set the stick postions at num_cates + 1 positions
        # ax.set_xticks(np.linspace(0, num_cates, num_cates + 1))
        # ax.set_yticks(np.linspace(0, num_cates, num_cates + 1))
        # # set the tick labels
        # ax.set_xticklabels(
        #     [
        #         f"{i}%" if i != 0 else "Top"
        #         for i in np.linspace(100, 0, num_cates + 1).astype(int)
        #     ]
        # )
        # ax.set_yticklabels(
        #     [
        #         f"{i}%" if i != 0 else ""
        #         for i in np.linspace(100, 0, num_cates + 1).astype(int)
        #     ]
        # )
        # # ax.set_yticklabels(
        # #     [
        # #         (f"{i}%" if i != 0 else "Top") if i != 100 else ""
        # #         for i in np.linspace(0, 100, num_cates + 1).astype(int)
        # #     ]
        # # )

        # # fig, ax = plt.subplots(figsize=(8, 8))
        # # im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        # # ax.figure.colorbar(im, ax=ax)
        # # ax.set(
        # #     xticks=np.arange(cm.shape[1]),
        # #     yticks=np.arange(cm.shape[0]),
        # #     xticklabels=np.unique(labels),
        # #     yticklabels=np.unique(labels),
        # #     title="Confusion matrix",
        # #     ylabel="True label",
        # #     xlabel="Predicted label",
        # # )
        # # # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # # # Loop over data dimensions and create text annotations.
        # # fmt = ".2f"
        # # thresh = cm.max() / 2.0
        # # for i in range(cm.shape[0]):
        # #     for j in range(cm.shape[1]):
        # #         ax.text(
        # #             j,
        # #             i,
        # #             format(cm[i, j], fmt),
        # #             ha="center",
        # #             va="center",
        # #             color="white" if cm[i, j] > thresh else "black",
        # #         )
        # # fig.tight_layout()

        # self.stratified_class_results["confusion matrix"] = cm
        # self.stratified_class_results["confusion matrix fig"] = fig

        # # save the confusion matrix
        # fig.savefig(
        #     os.path.join(
        #         self.log_dir,
        #         f"confusion_matrix_{use_set}_{self.config['task_name']}.png",
        #     )
        # )

        # fig.savefig(
        #     os.path.join(
        #         self.log_dir,
        #         f"confusion_matrix_{use_set}_{self.config['task_name']}.svg",
        #     ),
        #     format="svg",
        # )

    def visualize(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray = None,
        predictions: np.ndarray = None,
        color_key: str = "labels",
    ) -> matplotlib.figure.Figure:
        """
        Visualize the embeddings with UMAP

        Args:
            embeddings (np.ndarray): Raw embeddings to visualize
            labels (np.ndarray, optional): Labels of the embeddings
            predictions (np.ndarray, optional): Predictions values. Defaults to None.
            color_key (str): Use which field to color the points. Defaults to "labels".

        Returns:
            matplotlib.figure.Figure
        """
        if color_key == "labels":
            color = labels
            legend_name = "Efficiency"
        elif color_key == "predictions":
            color = predictions
            legend_name = "Predicted efficiency"

        if labels is not None and predictions is not None:
            self._eval_stratified_classes(labels, predictions)

        if predictions is not None and labels is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        else:
            fig, ax1 = plt.subplots(figsize=(12, 10))

        # umap visualization of embeddings with hue as labels
        # reducer = umap.UMAP(n_neighbors=30, min_dist=1.0, spread=1.0)
        # reducer = umap.UMAP(n_neighbors=60, min_dist=1.0, spread=1.0)
        # reducer = umap.UMAP(n_neighbors=60, min_dist=1.0, spread=1.0, metric="cosine")
        # reducer = umap.UMAP(
        #     n_neighbors=60,
        #     min_dist=1.0,
        #     spread=1.0,
        #     metric="cosine",
        #     random_state=12,
        # )
        # embedding = reducer.fit_transform(embeddings)

        # self.umap_emb = embedding
        # # customize cmap
        # # cmap_ = matplotlib.cm.get_cmap("Accent_r")
        # # make a cmap using three key colors
        # cmap_ = matplotlib.colors.LinearSegmentedColormap.from_list(
        #     "mycmap",
        #     # ["#44548c", "#5a448e", "#377280"],
        #     # ["#5f6da0", "#735fa2", "#518692"],
        #     # ["#5a448e", "#735fa2", "#cec6e1", "#bed5db", "#377280"],
        #     # ["#5a448e", "#907fb7", "#709d7a", "#377280"],
        #     ["#cec6e1", "#bda3cd", "#907fb7", "#735fa2", "#5a448e"],
        # )
        # im = ax1.scatter(
        #     embedding[:, 0],
        #     embedding[:, 1],
        #     c=color,
        #     s=60 * 1000 / len(color),
        #     cmap=cmap_,
        #     alpha=0.9,
        # )
        # # remove ticks and spines
        # ax1.set(xticks=[], yticks=[])
        # ax1.set_title("UMAP projection of the dataset", fontsize=24)
        # ax1.spines["top"].set_visible(False)
        # ax1.spines["right"].set_visible(False)
        # ax1.spines["bottom"].set_visible(False)
        # ax1.spines["left"].set_visible(False)
        # cbar = fig.colorbar(
        #     im,
        #     ax=ax1,
        #     # ticks=[i for i in np.linspace(np.min(color), np.max(color), 5)],
        #     format="%.1f",
        #     orientation="vertical",
        #     shrink=0.5,
        # )
        # cbar.ax.tick_params(labelsize=16)
        # cbar.ax.set_ylabel(legend_name, rotation=270, fontsize=16, labelpad=20)

        # correlation scatter plot bettwen labels and predictions
        if predictions is not None and labels is not None:
            ax2.scatter(
                labels,
                predictions,
                s=60 * 1000 / len(color),
                alpha=0.7,
            )
            ax2.set_title("Correlation between labels and predictions", fontsize=24)
            ax2.set_xlabel("Labels", fontsize=16)
            ax2.set_ylabel("Predictions", fontsize=16)
            ax2.tick_params(axis="both", which="major", labelsize=16)
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.set_xlim([np.min(labels), np.max(labels)])
            ax2.set_ylim([np.min(predictions), np.max(predictions)])
            ax2.plot(
                [np.min(labels), np.max(labels)],
                [np.min(predictions), np.max(predictions)],
                "r--",
            )

            # write the correlation coefficient on the plot
            corr = pearsonr(labels, predictions)[0]
            ax2.text(
                0.7,
                0.3,
                f"Corr: {corr:.2f}",
                transform=ax2.transAxes,
                verticalalignment="top",
                fontsize=16,
                color="red",
            )

        return fig

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join(
                "./finetune", self.config["model_to_evaluate"], "checkpoints"
            )
            state_dict = torch.load(
                os.path.join(checkpoints_folder, "model.pth"), map_location=self.device
            )
            # model.load_state_dict(state_dict)
            model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model



if __name__ == "__main__":
    import argparse
    import random

    seed = 0

    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    torch.manual_seed(seed = 0)
    torch.cuda.manual_seed(seed = 0)
    torch.cuda.manual_seed_all(seed = 0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Name of the checkpoint")
    args = parser.parse_args()
    # args = parser.parse_args(["Copy_Nov17_23-19-18_lnp_hela_with_feat_expt_Hela"])
    config_file = os.path.join(
        "./finetune", args.checkpoint, "checkpoints", "config_finetune.yaml"
    )
    config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
    config["model_to_evaluate"] = args.checkpoint

    if config["task_name"] == "lnp_hela_with_feat":
        config["dataset"]["task"] = "regression"
        config["dataset"][
            "data_path"
        ] = "data/finetuning_set_smiles_plus_features.csv"
        target_list = ["expt_Hela"]
        config["dataset"]["feature_cols"] = get_desc_cols(
            config["dataset"]["data_path"]
        )
        config["model"]["pred_additional_feat_dim"] = len(
            config["dataset"]["feature_cols"]
        )

    elif config["task_name"] == "lnp_raw_with_feat":
        config["dataset"]["task"] = "regression"
        config["dataset"][
            "data_path"
        ] = "data/finetuning_set_smiles_plus_features.csv"
        target_list = ["expt_Raw"]
        config["dataset"]["feature_cols"] = get_desc_cols(
            config["dataset"]["data_path"]
        )
        config["model"]["pred_additional_feat_dim"] = len(
            config["dataset"]["feature_cols"]
        )

    elif config["task_name"] == "smiles12000_with_feat":
        config["dataset"]["task"] = "regression"
        config["dataset"][
            "data_path"
        ] = "data/candidate_set_smiles_plus_features.csv"
        target_list = ["desc_ABC/10"]
        config["dataset"]["feature_cols"] = get_desc_cols(
            config["dataset"]["data_path"]
        )
        config["model"]["pred_additional_feat_dim"] = len(
            config["dataset"]["feature_cols"]
        )
        config["headtail_label_file"] = ""

    else:
        raise ValueError("Undefined downstream task!")

    print(config)

    results_list = []
    for target in target_list:
        config["dataset"]["target"] = target
        # result = main(config)
        # results_list.append([target, result])
        dataset = MolTestDatasetWrapper(config["batch_size"], **config["dataset"])

        infer_agent = Inference(dataset, config)
        pred, embs, labels = infer_agent.inference()
        results_list.append(
            {
                "result_rmse": getattr(infer_agent, "rmse", None),
                "result_corr": getattr(infer_agent, "corr", None),
            }
        )

        if target.startswith("expt_"):
            # experiment data with labels
            fig = infer_agent.visualize(embs, labels, predictions=pred)
        else:
            fig = infer_agent.visualize(embs, predictions=pred, color_key="predictions")


