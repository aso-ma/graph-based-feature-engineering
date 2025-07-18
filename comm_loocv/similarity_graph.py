import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm 
import pickle
from constants import Constants
from os import path

def __euclidean_distance(a: pd.Series, b: pd.Series) -> float:
    if len(a) != len(b):
        raise ValueError("Input Series must have the same length to calculate Euclidean distance.")
    squared_diff = (a - b)**2
    distance = np.sqrt(squared_diff.sum())

    return float(distance)

def generate_similarity_graph(dataframe: pd.DataFrame, graph_name: str, save_graph_file: bool = False) -> nx.Graph:
    edges = []
    for i_idx in dataframe.index:
        a = dataframe.loc[i_idx]
        for j_idx in dataframe.index:
            if j_idx > i_idx:
                b = dataframe.loc[j_idx]
                dist = __euclidean_distance(a, b)
                # inverse similarity
                similarity = 1 / (1 + dist)
                edges.append((i_idx, j_idx, similarity))
                
    similarity_graph = nx.Graph()
    similarity_graph.add_weighted_edges_from(edges)
    
    if save_graph_file:
        with open(Constants.GRAPH_DIR_PATH + graph_name + '.pkl', 'wb') as f:
            pickle.dump(similarity_graph, f)
    return similarity_graph

# def get_similarity_graph(dataframe: pd.DataFrame) -> nx.Graph:
#     if not path.exists(Constants.SIMILARITY_GRAPH_PATH):
#         __generate_similarity_graph(dataframe)
#     with open(Constants.SIMILARITY_GRAPH_PATH, 'rb') as f:
#         similarity_graph: nx.Graph = pickle.load(f)

#     return similarity_graph

def edge_removal(graph: nx.Graph, th: float, a_node: int | str | None = None) -> nx.Graph:
    temp_graph = graph.copy()
    if a_node is None:
        # global weak edges
        weak_edges = [(u, v) for u, v, attr in temp_graph.edges(data=True) if attr.get('weight', 0) < th]
    else:
        if not a_node in temp_graph:
            raise ValueError("One or more nodes in the list are not in the graph.")
        # local weak edges
        weak_edges = [(u, v) for u, v, attr in temp_graph.edges(a_node, data=True) if attr.get('weight', 0) < th]
    temp_graph.remove_edges_from(weak_edges)
    return temp_graph

def add_test_node_to(
    train_graph: nx.Graph, 
    X_train: pd.DataFrame,
    test_node: int | str, 
    test_node_feat: pd.Series
) -> nx.Graph:
    test_graph = train_graph.copy()
    test_graph.add_node(test_node)
    test_edges = []
    for node_j in X_train.index:
        train_series = X_train.loc[node_j]
        distance = __euclidean_distance(test_node_feat, train_series)
        similarity = 1 / (1 + distance) 
        test_edges.append((test_node, node_j, similarity))
    test_graph.add_weighted_edges_from(test_edges)
    test_graph = edge_removal(test_graph, Constants.EDGE_RM_TH, test_node)
    if test_graph.degree(nbunch=test_node) == 0:
        sorted_edges = sorted(test_edges, key=lambda x: x[2], reverse=True)
        percentage_to_take = 1 - Constants.EDGE_RM_TH
        n_items = int(len(sorted_edges) * percentage_to_take)
        n_items = max(1, n_items)
        top_edges = test_edges[:n_items]
        test_graph.add_weighted_edges_from(top_edges)
    return test_graph
