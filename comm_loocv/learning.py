import pandas as pd
from typing import List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tqdm import tqdm
from constants import Constants
import numpy as np
from community_detection import CD_METHODS

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

def __largest_power_of_2_less_than(n):
    if n <= 1:
        return None 
    return 1 << (n.bit_length() - 1)

def __get_num_principal_component(graph_augmented_data: pd.DataFrame) -> int: 
    count = 0
    for meth in CD_METHODS.keys():
        prefix = f"{meth}_comm_"
        count += len([col for col in graph_augmented_data.columns if col.startswith(prefix)])
    pow_of_2 = __largest_power_of_2_less_than(count)
    return max(2, pow_of_2 or count)

def perform_pca(graph_augmented_train_data: pd.DataFrame) -> Tuple[pd.DataFrame, PCA]:
    if Constants.ENCODING_METHOD != "one_hot":
        raise ValueError(f"Invalid encoding_method: {Constants.ENCODING_METHOD}. Must be 'one_hot'.")

    num_pc = __get_num_principal_component(graph_augmented_train_data)
    pca = PCA(n_components=num_pc)
    principal_components = pca.fit_transform(graph_augmented_train_data)
    df_pca_result = pd.DataFrame(
        index=graph_augmented_train_data.index,
        data=principal_components,
        columns=[f'PC{i}' for i in range(1, num_pc + 1)]
    )
    return df_pca_result, pca

def transform_test_pca(graph_augmented_test_data: pd.DataFrame, fitted_pca: PCA) -> pd.DataFrame:
    if Constants.ENCODING_METHOD != "one_hot":
        raise ValueError(f"Invalid encoding_method: {Constants.ENCODING_METHOD}. Must be 'one_hot'.")
    num_pc = __get_num_principal_component(graph_augmented_test_data)
    principal_components = fitted_pca.transform(graph_augmented_test_data)
    df_pca_result = pd.DataFrame(
        index=graph_augmented_test_data.index,
        data=principal_components,
        columns=[f'PC{i}' for i in range(1, num_pc + 1)]
    )
    return df_pca_result