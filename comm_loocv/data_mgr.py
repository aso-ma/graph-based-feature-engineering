import pandas as pd
from typing import List, Tuple
from os import path
import pickle
from constants import Constants
from sklearn.model_selection import LeaveOneOut

def preprocess(dataframe: pd.DataFrame) -> pd.DataFrame:
    # do some preprocessing if required 
    return dataframe

def __loocv_split(dataframe: pd.DataFrame) -> None:
    X = dataframe.drop(Constants.TARGET_COL, axis=1)
    y = dataframe[Constants.TARGET_COL]

    fold_list = []
    loo = LeaveOneOut()
    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        fold_list.append((X_train, X_test, y_train, y_test))

    with open(Constants.FOLD_DATA_PATH, 'wb') as f:
       pickle.dump(fold_list, f)

def get_fold_data(dataframe: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    if not path.exists(Constants.FOLD_DATA_PATH):
        __loocv_split(dataframe)
    with open(Constants.FOLD_DATA_PATH, 'rb') as f:
        fold_data = pickle.load(f)
    return fold_data