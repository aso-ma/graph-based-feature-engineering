from dataclasses import dataclass
from typing import List, Tuple, Final, Dict, Union
from sklearn.base import ClassifierMixin
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.base import clone
import gower 
import networkx as nx
from scipy.stats import ttest_ind
import pickle
from cdlib import algorithms
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class Constants:
    DATA_PATH: Final[str] = "./data/tabular_data.csv"
    TARGET_COL: Final[str] = "Target"
    OUT_DIR_PATH: Final[str] = "./output/"
    FOLD_COUNT: Final[int] = 10

    # List of classifiers
    CLFs = [
        RandomForestClassifier(), 
        LogisticRegression(max_iter=250), 
        KNeighborsClassifier(n_neighbors=3)
    ]
    
    # List of evaluation metrics
    EVAL_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'matthews_corrcoef']

    # Edge Removal Threshold
    EDGE_RM_TH = 0.7

    # Statistic test significance level
    SIGNIFICANCE_LEVEL = 0.9

    # Community detection methods
    COMM_DET_METHODS = ['louvain', 'girvan_newman']

    # Number of principal components
    NUM_PC = 2


class StatsFeatSel:
    _significant_cols = []
    _significant_graph_cols = set()

    @staticmethod
    def comp_significant_feat(dataframe: pd.DataFrame):
        print("Statistically Feature Selection")
        graph = generate_similarity_graph(dataframe)
        graph = edge_rm(graph, Constants.EDGE_RM_TH)
        df_graph_feat = extract_graph_features(graph)
        df_concat = concatenate_dataframes(dataframe, df_graph_feat)

        groups = df_concat[Constants.TARGET_COL].unique()
        # Ensure the target column is binary
        if len(groups) != 2:
            raise ValueError("Target column must be binary for t-test.")
        # Split the data into two groups based on the target column
        group1 = df_concat[df_concat[Constants.TARGET_COL] == groups[0]]
        group2 = df_concat[df_concat[Constants.TARGET_COL] == groups[1]]
        for col in df_concat.columns:
            if col != Constants.TARGET_COL:
                t_stat, p_value = ttest_ind(group1[col], group2[col], nan_policy='omit')
                if p_value < Constants.SIGNIFICANCE_LEVEL:
                    if col.startswith(tuple(Constants.COMM_DET_METHODS)):
                        comm_method = next((method for method in Constants.COMM_DET_METHODS if col.startswith(method)), None)
                        StatsFeatSel._significant_graph_cols.add(comm_method)
                    else:
                        StatsFeatSel._significant_cols.append(col)

        with open(Constants.OUT_DIR_PATH + 'significant_columns.pkl', 'wb') as f:
            pickle.dump(StatsFeatSel._significant_cols, f)
        with open(Constants.OUT_DIR_PATH + 'significant_graph_columns.pkl', 'wb') as f:
            pickle.dump(StatsFeatSel._significant_graph_cols, f)

    @staticmethod
    def get_features():
        return StatsFeatSel._significant_cols, StatsFeatSel._significant_graph_cols

def concatenate_dataframes(df_primary_feature, df_graph_feature):
    return pd.concat([df_primary_feature, df_graph_feature], axis=1)

def preprocess(dataframe: pd.DataFrame) -> pd.DataFrame:
    # do some preprocessing if required 
    return dataframe

def k_fold_split(dataframe: pd.DataFrame, k: int) -> List[Tuple[int, pd.DataFrame, pd.DataFrame]]:
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    folds = []
    for fold_num, (train_index, test_index) in enumerate(kf.split(dataframe), start=1):
        train = dataframe.iloc[train_index]
        test = dataframe.iloc[test_index]
        folds.append((fold_num, train, test))
    return folds 

def primary_evaluation(fold_data: List[Tuple[int, pd.DataFrame, pd.DataFrame]], result_file_name: str):    
    df_result = pd.DataFrame(index=Constants.EVAL_METRICS, columns=[clf.__class__.__name__ for clf in Constants.CLFs])
    for m in Constants.EVAL_METRICS:
        for clf in Constants.CLFs:
            df_result.at[m, clf.__class__.__name__] = []

    for fold_num, df_train, df_test in tqdm(fold_data, desc="Primary Classification"):
        X_train = df_train.drop(columns=[Constants.TARGET_COL])
        y_train = df_train[Constants.TARGET_COL]
        X_test = df_test.drop(columns=[Constants.TARGET_COL])
        y_test = df_test[Constants.TARGET_COL]

        fold_results = {}
        for clf in Constants.CLFs:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_pred_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
            
            fold_results[clf.__class__.__name__] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else None,
                'matthews_corrcoef': matthews_corrcoef(y_test, y_pred)
            }
        # Aggregate results for each fold
        for clf_name, scores in fold_results.items():
            for metric, value in scores.items():
                df_result.at[metric, clf_name].append(value)

    # Average the results across folds
    df_result = df_result.map(lambda x: sum(x) / len(x) if isinstance(x, list) else x)
    # Save the result DataFrame to a pickle file
    df_result.to_pickle(Constants.OUT_DIR_PATH + result_file_name + '.pkl')
    
def generate_similarity_graph(dataframe: pd.DataFrame) -> nx.Graph:
    df_feature = dataframe.drop(columns=[Constants.TARGET_COL])
    distance_matrix = gower.gower_matrix(df_feature)
    similarity_matrix = 1 - distance_matrix 
    

    similarity_graph = nx.Graph()
    similarity_graph.add_nodes_from(df_feature.index)
    
    for i, node_i in enumerate(df_feature.index):
        for j, node_j in enumerate(df_feature.index):
            if i < j:
                similarity_graph.add_edge(node_i, node_j, weight=similarity_matrix[i, j])
    return similarity_graph


def edge_rm(graph: nx.Graph, th: float, nodes: Union[List[str], List[int], None] = None) -> nx.Graph:

    if nodes is None:
        # global weak edges 
        weak_edges = [(u, v) for u, v, attr in graph.edges(data=True) if attr.get('weight', 0) < th]
    else:
        if not all(node in graph for node in nodes):
            raise ValueError("One or more nodes in the list are not in the graph.")
        # local weak edges
        weak_edges = [(u, v) for u, v, attr in graph.edges(nodes, data=True) if attr.get('weight', 0) < th]
    graph.remove_edges_from(weak_edges)
    return graph

def extract_graph_features(graph: nx.Graph, only_significant: bool = False) -> pd.DataFrame:
    graph_feat_list = []
    if (only_significant):
        _, g_significant_feats = StatsFeatSel.get_features()
        graph_feat_list = g_significant_feats
    else: 
        graph_feat_list = Constants.COMM_DET_METHODS

    df_graph_features = pd.DataFrame(index=graph.nodes())

    if 'louvain' in graph_feat_list:
        communities = algorithms.louvain(graph)
        df_graph_features = insert_community_columns_into_df(df_graph_features, communities, 'louvain')

    if 'girvan_newman' in graph_feat_list:
        communities = algorithms.girvan_newman(graph, level=3)
        df_graph_features = insert_community_columns_into_df(df_graph_features, communities, 'girvan_newman')

    return df_graph_features

# def extract_graph_features_for(graph: nx.Graph, test_nodes: Union[List[str], List[int]]) -> pd.DataFrame:
#     if not all(node in graph for node in test_nodes):
#         raise ValueError("Not all nodes in the list are in the graph!")
    
#     df_features = extract_graph_features(graph, True)
#     return df_features

def insert_community_columns_into_df(dataframe: pd.DataFrame, comms: List[List[int]], comm_label: str) -> pd.DataFrame:
    for idx, community in enumerate(comms.communities, start=1):
        column_name = f'{comm_label}_comm_{idx}'
        # Set values to 1 if node is in the community, otherwise 0
        dataframe[column_name] = [1 if node in community else 0 for node in dataframe.index]
    
    return dataframe

def perform_pca(dataframe: pd.DataFrame) -> pd.DataFrame:
    pca = PCA(n_components=Constants.NUM_PC)
    principal_components = pca.fit_transform(dataframe)
    df_pca_result = pd.DataFrame(
        index=dataframe.index, 
        data=principal_components, 
        columns=[f'PC{i}' for i in range(1, Constants.NUM_PC + 1)]
    )
    return df_pca_result

def train_clfs_for(df_train: pd.DataFrame) -> Dict[str, ClassifierMixin]:
    trained_clfs = dict()
    X_train = df_train.drop(columns=[Constants.TARGET_COL])
    y_train = df_train[Constants.TARGET_COL]
    for clf in Constants.CLFs:
        clf_instance = clone(clf) 
        clf_instance.fit(X_train, y_train)
        trained_clfs[clf.__class__.__name__] = clf_instance

    return trained_clfs

def add_test_nodes(graph: nx.Graph, test_nodes: Union[List[str], List[int]], 
                   df_train: pd.DataFrame, df_test: pd.DataFrame) -> nx.Graph:
    df_train_featurs = df_train.drop(columns=[Constants.TARGET_COL])
    df_test_features = df_test.drop(columns=[Constants.TARGET_COL])
    distances = gower.gower_matrix(df_test_features, df_train_featurs)
    similarities = 1 - distances

    edges = []    
    for i, node_i in enumerate(df_test_features.index):
        for j, node_j in enumerate(df_train_featurs.index):
            edges.append((node_i, node_j, similarities[i, j]))

    graph.add_weighted_edges_from(edges)
    
    test_graph = edge_rm(graph, Constants.EDGE_RM_TH, test_nodes)
    return test_graph

def fold_evaluation(df_final_evaluation: pd.DataFrame, trained_clfs: Dict[str, ClassifierMixin], df_test: pd.DataFrame, result_file_name:     str) -> pd.DataFrame:
    df_test[Constants.TARGET_COL] = df_test[Constants.TARGET_COL].astype('int')
    X_test = df_test.drop(columns=[Constants.TARGET_COL])
    y_test = df_test[Constants.TARGET_COL]
    df_result = pd.DataFrame(index=Constants.EVAL_METRICS, columns=trained_clfs.keys())
    for name, clf in trained_clfs.items():
        y_pred = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else None
        mcc = matthews_corrcoef(y_test, y_pred)

        df_result.at['accuracy', name] = accuracy
        df_result.at['precision', name] = precision
        df_result.at['recall', name] = recall
        df_result.at['f1', name] = f1
        df_result.at['roc_auc', name] = auc
        df_result.at['matthews_corrcoef', name] = mcc

        df_final_evaluation.at['accuracy', name].append(accuracy)
        df_final_evaluation.at['precision', name].append(precision)
        df_final_evaluation.at['recall', name].append(recall)
        df_final_evaluation.at['f1', name].append(f1)
        df_final_evaluation.at['roc_auc', name].append(auc)
        df_final_evaluation.at['matthews_corrcoef', name].append(mcc)

    df_result.to_pickle(Constants.OUT_DIR_PATH + result_file_name + '.pkl')

if __name__ == "__main__":
    # load data
    data = pd.read_csv(Constants.DATA_PATH)
    # preprocess 
    data = preprocess(data)

    # find significant features 
    StatsFeatSel.comp_significant_feat(data)
    significant_feat, significant_graph_feat = StatsFeatSel.get_features()
    if (len(significant_feat) == 0):
        raise ValueError("There are no statistically significant features in the data!")
    if (len(significant_feat) == 0):
        raise ValueError("There are no statistically significant graph-based features in the data!")

    # split data into k-folds
    fold_data = k_fold_split(data, Constants.FOLD_COUNT)
    # do a classification task with primary data
    primary_evaluation(fold_data, 'primary_classification_result')
    # define a dataframe for final evaluation result
    df_final_evaluation = pd.DataFrame(index=Constants.EVAL_METRICS, columns=[clf.__class__.__name__ for clf in Constants.CLFs])
    for m in Constants.EVAL_METRICS:
        for clf in Constants.CLFs:
            df_final_evaluation.at[m, clf.__class__.__name__] = []
    
    # iterate through folds
    for fold_num, df_train, df_test in tqdm(fold_data, desc="Processing Folds"):
        # create a complete similarity graph from train data
        g_train_complete = generate_similarity_graph(df_train)
        # weak edge removal
        g_train = edge_rm(g_train_complete, Constants.EDGE_RM_TH)
        # extract significant graph-based features from similarity graph
        df_train_graph_feat = extract_graph_features(g_train, True)
        # perform pca 
        df_train_graph_pca = perform_pca(df_train_graph_feat)
        # concatenate graph-based features with primary features
        df_train_final = concatenate_dataframes(df_train, df_train_graph_pca)
        # train classifiers over primary features + graph-based features
        trained_clfs = train_clfs_for(df_train_final)
        # prepare test data by extracting graph-based
        g_test = add_test_nodes(g_train, df_test.index.values, df_train, df_test)
        df_test_graph_feat = extract_graph_features(g_test, True)
        df_test_nodes_feat = df_test_graph_feat.loc[df_test.index.values]
        df_test_nodes_pca = perform_pca(df_test_nodes_feat)
        df_test_nodes_final = concatenate_dataframes(df_test, df_test_nodes_pca)
        # remove the test nodes from graph
        g_test.remove_nodes_from(df_test.index.values)
        g_train = g_test
        # test 
        fold_evaluation(df_final_evaluation, trained_clfs, df_test_nodes_final, f'final_classification_result_fold_{fold_num}')

    # Average the results across folds
    df_final_evaluation = df_final_evaluation.map(lambda x: sum(x) / len(x) if isinstance(x, list) else x)
    # Save the result DataFrame to a pickle file
    df_final_evaluation.to_pickle(Constants.OUT_DIR_PATH  + 'df_final_evaluation.pkl')    



