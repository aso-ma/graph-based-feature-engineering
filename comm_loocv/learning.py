import pandas as pd
from typing import List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tqdm import tqdm
from constants import Constants
import numpy as np

CLFs = [
    RandomForestClassifier(), 
    KNeighborsClassifier(n_neighbors=3)
]

EVAL_METRICS = [accuracy_score, f1_score, recall_score]

def evaluate_classifiers(fold_data: List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]], 
    primary: bool = True
) -> None:
    
    saving_path = Constants.PRIMARY_EVAL_RESULT_PATH if primary else Constants.SECONDARY_EVAL_RESULT_PATH

    df_result = pd.DataFrame(
        index=[metric.__name__ for metric in EVAL_METRICS], 
        columns=[clf.__class__.__name__ for clf in CLFs]
    )
    
    for clf in CLFs:
        print(clf.__class__.__name__)
        y_true, y_pred = [], []
        for X_train, X_test, y_train, y_test  in tqdm(fold_data, desc="Folds"):
            X_train = np.asarray(X_train)
            X_test = np.asarray(X_test)
            y_train = np.asarray(y_train)
            y_test = np.asarray(y_test)

            clf.fit(X_train, y_train)
            y_pred.append(clf.predict(X_test)[0])
            y_true.append(y_test[0])   

        for metric in EVAL_METRICS:
            score = metric(y_true, y_pred)
            df_result.at[metric.__name__, clf.__class__.__name__] = round(score, 5)
    
    df_result.to_pickle(saving_path)

def perform_pca(train_data: pd.DataFrame) -> Tuple[pd.DataFrame, PCA]:
    pca = PCA(n_components=Constants.NUM_PC)
    principal_components = pca.fit_transform(train_data)
    df_pca_result = pd.DataFrame(
        index=train_data.index,
        data=principal_components,
        columns=[f'PC{i}' for i in range(1, Constants.NUM_PC + 1)]
    )
    return df_pca_result, pca

def transform_test_pca(test_data: pd.DataFrame, fitted_pca: PCA) -> pd.DataFrame:
    principal_components = fitted_pca.transform(test_data)
    df_pca_result = pd.DataFrame(
        index=test_data.index,
        data=principal_components,
        columns=[f'PC{i}' for i in range(1, Constants.NUM_PC + 1)]
    )
    return df_pca_result