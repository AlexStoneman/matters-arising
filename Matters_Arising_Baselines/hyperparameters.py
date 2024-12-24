
import numpy as np
import itertools
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from MLP_architecture import *

XGBoost_hypers = {
        'n_estimators': [50, 100, 250],
        'eta': [0.1, 0.25, 0.5, 1.0],
        'subsample': [0.3, 0.5, 0.7, 0.9],
        'objective': ['reg:squarederror'],
        'eval_metric': ["rmse"]
        }

RF_hypers = {'n_estimators': [50, 100, 250, 500],       
             'max_depth': [10, 30, 50]
        }   

MLP_hypers = {'eta': [0.001, 0.005],       
             'num_layers': [3,5],
             'layer_depth': [256,512]
        }   


