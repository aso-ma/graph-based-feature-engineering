from typing import Dict, Callable
from cdlib import algorithms
import networkx as nx
import pandas as pd
from functools import partial
from warnings import warn

CD_METHODS: Dict[str, Callable] = {
    'louvain': partial(algorithms.louvain),
    'girvan_newman': partial(algorithms.girvan_newman, level=3)
}

def extract_graph_features(graph: nx.Graph) -> tuple[pd.DataFrame, dict]:
    df_graph_features = pd.DataFrame(index=graph.nodes())
    community_result = {}
    for method_name, method_func in CD_METHODS.items():
        communities_dict = {}
        communities = method_func(graph)
        num_comms = len(communities.communities)
        if num_comms <= 1:
            warn(f"The <{method_name}> algorithm was excluded from further analysis as it identified only a single community.", UserWarning)    
            continue

        for idx, community in enumerate(communities.communities, start=1):
            column_name = f"{method_name}_comm_{idx}"
            # Set values to 1 if node is in the community, otherwise 0
            df_graph_features[column_name] = [1 if node in community else 0 for node in df_graph_features.index]
            communities_dict[idx] = community
        
        community_result[method_name] = communities_dict
    
    return df_graph_features, community_result
        
def assign_by_neighbors(test_graph: nx.Graph, test_node: str|int, community_result: dict) -> dict:
    """Assign node to community by neighbors
    """
    output_dict = {}
    node_neighbors = list(test_graph.neighbors(n=test_node))
    for method_name, communities_dict in community_result.items():
        neighbor_communities = {comm_idx: 0 for comm_idx in communities_dict.keys()}
        for comm_idx, community in communities_dict.items():
            for n in node_neighbors:
                if n in community:
                    neighbor_communities[comm_idx] += 1
        comm_with_max_value = max(neighbor_communities, key=neighbor_communities.get)
        output_dict[method_name] = comm_with_max_value

    return output_dict

def assign_by_modularity(test_graph: nx.Graph, test_node: str|int, community_result: dict) -> dict:
    """ Assign node to community by modularity
    """
    output_dict = {}
    for method_name, communities_dict in community_result.items():
        meth_mod_dict = {comm_idx: 0.0 for comm_idx in communities_dict.keys()}
        for comm_idx, community in communities_dict.items():
            communities = [comm for cid, comm in communities_dict.items() if cid != comm_idx]
            comm_list = community[:]
            comm_list.append(test_node)
            communities.append(comm_list)
            meth_mod_dict[comm_idx] = nx.community.modularity(G=test_graph, communities=communities)
            pass
        comm_with_max_value = max(meth_mod_dict, key=meth_mod_dict.get)
        output_dict[method_name] = comm_with_max_value    
    return output_dict

def get_community_features_for(test_node: str|int, node_meth_comm: dict, community_result: dict) -> pd.DataFrame:
    columns = []
    for meth, communities_dict in community_result.items():
        num_comm = len(communities_dict)
        for idx in range(1, num_comm + 1):
            columns.append(f"{meth}_comm_{idx}")
    df = pd.DataFrame(index=[test_node], columns=columns)
    for meth, communities_dict in community_result.items():
        num_comm = len(communities_dict)
        comm_idx = node_meth_comm[meth]
        for idx in range(1, num_comm + 1):
            col_name = f"{meth}_comm_{idx}"
            if idx == comm_idx:
                df.at[test_node, col_name] = 1
            else:
                df.at[test_node, col_name] = 0
    return df