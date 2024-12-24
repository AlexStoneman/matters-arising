import numpy as np
from xgboost import XGBRegressor
from MLP_architecture import *
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import joblib
import os 
import math

from splits import *
from hyperparameters import *
from featurization import *


def train_test_ensemble(run_name, model_name, features, cell, split, num_iterations, gpu):

    RMSEs = np.zeros((num_iterations,))
    for i in range(num_iterations):
        
        results_dir = os.path.join(run_name,"iter"+str(i))
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        seed = i
        library_split_seed = i
        y_pred, y_true, _ = train_test_model(results_dir, model_name, features, cell, split, seed, library_split_seed, gpu)

        RMSEs[i] = root_mean_squared_error(y_pred, y_true)
        print("RMSE on Split "+str(i)+": "+str(RMSEs[i]) + "\n")
    
    print("Mean RMSE on all splits: " + str(np.mean(RMSEs)))


def make_dataset(features: str, cell: str, split: str, library_split_seed: int):

    if split == "murcko":
        train_idx = murcko_scaffold_indices["train"]
        val_idx = murcko_scaffold_indices["val"]
        test_idx = murcko_scaffold_indices["test"]
    elif split == "library":
        one_hot_mat = np.loadtxt("onehot_encoding.csv", delimiter=",")
        train_idx, val_idx, test_idx = get_library_scaffold_split(one_hot_mat, seed=library_split_seed)
    
    y_train, y_val, y_test = get_transfecton_efficiency(train_idx, val_idx, test_idx, cell)
    X_train, X_val, X_test = get_features(features, train_idx, val_idx, test_idx)
    dataset = LNP_Dataset(X_train, X_val, X_test, y_train, y_val, y_test)

    return dataset


def train_model(model_name: str, seed: int, hyper_set: dict, dataset, gpu, results_dir):
    if model_name == "XGB":
        model = XGBRegressor(random_state = seed, **hyper_set)
        model.fit(dataset.X_train, dataset.y_train)
    elif model_name == "RF":
        model = RandomForestRegressor(random_state = seed, **hyper_set)
        model.fit(dataset.X_train, dataset.y_train)
    elif model_name == "MLP":
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed = seed)
        torch.cuda.manual_seed_all(seed=seed)
        np.random.seed(seed=seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        model = MLP(seed = seed, input_dim = dataset.X_train.shape[1], num_layers=hyper_set["num_layers"], layer_depth=hyper_set["layer_depth"], eta=hyper_set["eta"],gpu=gpu,folder=results_dir)
        model = model.to(gpu)
        model.fit(dataset.X_train, dataset.y_train, dataset.X_val, dataset.y_val)
    
    return model


def find_best_hypers(model_name, seed, dataset, gpu = "cpu", results_dir = ""):

    if model_name == "XGB":
        hyper_options = XGBoost_hypers
    elif model_name == "RF":
        hyper_options = RF_hypers
    elif model_name == "MLP":
        hyper_options = MLP_hypers
    all_hypers = [dict(zip(hyper_options.keys(), v)) for v in itertools.product(*hyper_options.values())]

    best_rmse = np.inf
    best_hypers = None
    for hyper_set in all_hypers:
        cur_model = train_model(model_name, seed, hyper_set, dataset, gpu, results_dir)
        if model_name == "MLP":
            rmse = root_mean_squared_error(cur_model.predict(dataset.X_val).cpu(), dataset.y_val)
        else:
            rmse = root_mean_squared_error(cur_model.predict(dataset.X_val), dataset.y_val)
        if rmse < best_rmse: 
            best_hypers = hyper_set
            best_rmse = rmse

    return best_hypers


def save_model(results_dir: str, model_name: str, model, best_hypers):
    model_file_name = os.path.join(results_dir,"model.joblib")
    print("Saving model as "+model_file_name+"\n")

    if model_name == "XGB" or model_name == "RF":
        joblib.dump(model, model_file_name)
    elif model_name == "MLP":
        optimizer = torch.optim.Adam(model.parameters(), lr = best_hypers["eta"]) 
        torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, model_file_name)  


def save_results(results_dir, dataset, y_pred):
    np.savetxt(os.path.join(results_dir,"y_true.csv"), dataset.y_test, delimiter=',')
    np.savetxt(os.path.join(results_dir,"y_pred.csv"), y_pred, delimiter=",")
    results_df = pd.DataFrame(
        {
            "predictions": y_pred,
            "labels": dataset.y_test,
        }
    )
    results_df.to_csv(os.path.join(results_dir, "testset_preds.csv"), index=False)


def train_test_model(results_dir: str, model_name: str, features: str, cell: str, split: str, seed: int = 0, library_split_seed: int = 0, gpu: str = "cpu"):
    
    assert(split in ["murcko","library"])
    assert(model_name in ["XGB","RF","MLP"])
    assert(features in ["OH","FP","AGILEdesc"])
    assert(cell in ["HeLa","RAW"])

    dataset = make_dataset(features=features, cell=cell, split=split, library_split_seed=library_split_seed)

    best_hypers = find_best_hypers(model_name = model_name, seed=seed, dataset=dataset, gpu=gpu, results_dir=results_dir)

    model = train_model(model_name, seed, best_hypers, dataset, gpu, results_dir)

    if model_name == "MLP":
        y_pred = model.predict(dataset.X_test).cpu()
    else: 
        y_pred = model.predict(dataset.X_test)

    save_model(results_dir, model_name, model, best_hypers)
    save_results(results_dir, dataset, y_pred)

    return y_pred, dataset.y_test, model

