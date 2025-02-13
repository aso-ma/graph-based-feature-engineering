from dataclasses import dataclass
from typing import List, Tuple, Final, Dict
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
    SIGNIFICANCE_LEVEL = 0.05

    # All Graph-Based Features
    GRAPH_BASED_FEATURES = ['degree_centrality', 'node_strength']

class StatsFeatSel:
    _significant_cols = []
    _significant_graph_cols = []

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
                    (StatsFeatSel._significant_graph_cols if col in Constants.GRAPH_BASED_FEATURES else StatsFeatSel._significant_cols).append(col)
        with open(Constants.OUT_DIR_PATH + 'significant_columns.pkl', 'wb') as f:
            pickle.dump(StatsFeatSel._significant_cols, f)
        with open(Constants.OUT_DIR_PATH + 'significant_graph_columns.pkl', 'wb') as f:
            pickle.dump(StatsFeatSel._significant_graph_cols, f)

    @staticmethod
    def get_features():
        return StatsFeatSel._significant_cols, StatsFeatSel._significant_graph_cols

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


def edge_rm(graph: nx.Graph, th: float, node: str | int | None = None) -> nx.Graph:
    if node != None and node not in graph:
        raise ValueError(f"Node '{node}' not found in the graph.")
    if node is None:
        # global weak edges 
        weak_edges = [(u, v) for u, v, attr in graph.edges(data=True) if attr.get('weight', 0) < th]
    else:
        # local weak edges
        weak_edges = [(u, v) for u, v, attr in graph.edges(node, data=True) if attr.get('weight', 0) < th]
    graph.remove_edges_from(weak_edges)
    return graph

def extract_graph_features(graph: nx.Graph, only_significant: bool = False) -> pd.DataFrame:
    graph_feat_list = []
    if (only_significant):
        df_graph_features = pd.DataFrame(index=graph.nodes(), columns=graph_feat_list)
        _, g_significant_feats = StatsFeatSel.get_features()
        graph_feat_list = g_significant_feats
    else: 
        df_graph_features = pd.DataFrame(index=graph.nodes(), columns=Constants.GRAPH_BASED_FEATURES)
        graph_feat_list = Constants.GRAPH_BASED_FEATURES

    if 'degree_centrality' in graph_feat_list:
        degree_centrality = nx.degree_centrality(graph)
        df_graph_features['degree_centrality'] = degree_centrality

    if 'node_strength' in graph_feat_list:
        node_strength = dict(graph.degree(weight='weight'))
        df_graph_features['node_strength'] = node_strength

    return df_graph_features

def extract_graph_features_for(graph: nx.Graph, node: str | int) -> pd.Series:
    if node not in graph:
        raise ValueError(f"the node, {node}, is not in the graph.")

    # get significant graph-based features
    _, graph_feat_list = StatsFeatSel.get_features()
    # extract only significant features for a node
    series = pd.Series()
    if 'degree_centrality' in graph_feat_list:
        series['degree_centrality'] = graph.degree(node)  
    if 'node_strength' in graph_feat_list:
        series['node_strength'] = graph.degree(node, weight='weight')

    return series;

def concatenate_dataframes(df_primary_feature, df_graph_feature):
    return pd.concat([df_primary_feature, df_graph_feature], axis=1)

def train_clfs_for(df_train: pd.DataFrame) -> Dict[str, ClassifierMixin]:
    trained_clfs = dict()
    X_train = df_train.drop(columns=[Constants.TARGET_COL])
    y_train = df_train[Constants.TARGET_COL]
    for clf in Constants.CLFs:
        clf_instance = clone(clf) 
        clf_instance.fit(X_train, y_train)
        trained_clfs[clf.__class__.__name__] = clf_instance

    return trained_clfs

def add_test_node(df_train: pd.DataFrame, graph: nx.Graph, test_node: str | int, test_node_primary_feat: pd.Series):
    df_feature = df_train.drop(columns=[Constants.TARGET_COL])
    distances = gower.gower_matrix(test_node_primary_feat.to_frame().T, df_feature).flatten()
    similarities = 1 - distances

    edges = []
    for i, df_index in enumerate(df_train.index):
        edges.append((test_node, df_index, similarities[i]))

    graph.add_weighted_edges_from(edges)
    
    test_graph = edge_rm(graph, Constants.EDGE_RM_TH, test_node)
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
        df_feat = extract_graph_features(g_train, True)
        # concatenate graph-based features with primary features
        df_concat = concatenate_dataframes(df_train, df_feat)
        # train classifiers over primary features + graph-based features
        trained_clfs = train_clfs_for(df_concat)
        # prepare test data by extracting graph-based
        df_test_final = pd.DataFrame(index=df_test.index, columns=df_concat.columns)
        for test_node, primary_feat in df_test.iterrows():
            # drop traget from test node features
            tnode_primary_feat = primary_feat.drop(Constants.TARGET_COL)
            # add test node to the graph
            g_test = add_test_node(df_train, g_train, test_node, tnode_primary_feat)
            # extract graph-based features for the test node
            tnode_graph_feat = extract_graph_features_for(g_test, test_node)
            # concatenate test node graph-based features with test node primary features
            tnode_features = pd.concat([primary_feat, tnode_graph_feat])
            # append test node features to test data
            df_test_final.loc[test_node] = tnode_features    
            # remove the test node from graph
            g_test.remove_node(test_node)
            g_train = g_test
        
        # test phase
        fold_evaluation(df_final_evaluation, trained_clfs, df_test_final, f'final_classification_result_fold_{fold_num}')
    
    # Average the results across folds
    df_final_evaluation = df_final_evaluation.map(lambda x: sum(x) / len(x) if isinstance(x, list) else x)
    # Save the result DataFrame to a pickle file
    df_final_evaluation.to_pickle(Constants.OUT_DIR_PATH  + 'df_final_evaluation.pkl')
