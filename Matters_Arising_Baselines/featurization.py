import pandas as pd
import numpy as np
from sklearn import preprocessing
from rdkit import Chem
from molvs import standardize_smiles
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator



class ECFP:
    def __init__(self, smiles, radius, num_bits):
        self.mols = [Chem.MolFromSmiles(i) for i in smiles]
        self.smiles = smiles
        self.radius = radius
        self.num_bits = num_bits
        self.fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius,fpSize=num_bits)

    def mol2fp(self, mol):
        fp = self.fp_gen.GetFingerprint(mol)
        array = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, array)
        return array

    def compute_ECFP(self):
        arr = np.empty((0,self.num_bits), int).astype(int)
        for i in self.mols:
            fp = self.mol2fp(i)
            arr = np.vstack((arr, fp))
        return arr.astype(float)
    

def get_fp_descriptors(train_idx,val_idx,test_idx,bits,radius):
    agile_df = pd.read_csv("finetuning_set_smiles_plus_features.csv")
    train_smiles_standarized = [standardize_smiles(i) for i in agile_df['smiles'].values]
    fp_descriptor = ECFP(train_smiles_standarized,radius=radius,num_bits=bits)
    fp_mat = fp_descriptor.compute_ECFP()
    X_train = fp_mat[train_idx]
    X_val = fp_mat[val_idx]
    X_test = fp_mat[test_idx]
    return X_train, X_val, X_test


def get_agile_mordred_descriptors(train_idx,val_idx,test_idx):
    agile_df = pd.read_csv("finetuning_set_smiles_plus_features.csv")
    agile_df = agile_df.drop(["smiles","expt_Hela","expt_Raw"],axis=1)
    agile_mat = agile_df.to_numpy()
    X_train = agile_mat[train_idx]
    X_val = agile_mat[val_idx]
    X_test = agile_mat[test_idx]
    return X_train, X_val, X_test


def get_one_hot_descriptors(train_idx,val_idx,test_idx):
    one_hot_mat = np.loadtxt("onehot_encoding.csv", delimiter=',')
    X_train = one_hot_mat[train_idx]
    X_val = one_hot_mat[val_idx]
    X_test = one_hot_mat[test_idx]
    return X_train, X_val, X_test


def get_transfecton_efficiency(train_idx,val_idx,test_idx,cell):
    agile_df = pd.read_csv("finetuning_set_smiles_plus_features.csv")
    if cell == "HeLa":
        data = agile_df["expt_Hela"]
    elif cell == "RAW":
        data = agile_df["expt_Raw"]

    data_mat = data.to_numpy()
    y_train = data_mat[train_idx]
    y_val = data_mat[val_idx]
    y_test = data_mat[test_idx]
    return y_train, y_val, y_test


def get_features(features, train_idx, val_idx, test_idx):
    assert(features in ["OH","FP","AGILEdesc"])
    if features == "OH":
        X_train, X_val, X_test = get_one_hot_descriptors(train_idx, val_idx, test_idx)
    elif features == "FP":
        X_train, X_val, X_test = get_fp_descriptors(train_idx, val_idx, test_idx,bits=2048,radius=5) 
    elif features == "AGILEdesc":
        X_train, X_val, X_test = get_agile_mordred_descriptors(train_idx, val_idx, test_idx)

    return X_train, X_val, X_test