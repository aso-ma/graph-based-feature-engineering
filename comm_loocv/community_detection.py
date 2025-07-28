from typing import Dict, Callable, List, Literal
from cdlib import algorithms
import networkx as nx
import pandas as pd
from functools import partial
from warnings import warn

CD_METHODS: Dict[str, Callable] = {
    'louvain': partial(algorithms.louvain),
    'girvan_newman': partial(algorithms.girvan_newman, level=3)
}

CD_RESULT_TYPE = Dict[str, Dict[int, List[str | int]]]
ENCODING_TYPE = Literal["binary", "one_hot"]

def __one_hot_encoding(community_result: CD_RESULT_TYPE) -> pd.DataFrame:
    node_set = {node for community_list in community_result[list(CD_METHODS.keys())[0]] for node in community_list }
    df_features = pd.DataFrame(index=node_set)
    for meth, communities_dict in community_result.items():
        for comm_idx, community in communities_dict.items():
            column_name = f"{meth}_comm_{comm_idx}"
            # Set values to 1 if node is in the community, otherwise 0
            df_features[column_name] = [1 if node in community else 0 for node in df_features.index]
    return df_features

def __int_to_binary(num: int, length: int) -> str:
    return format(num, f'0{length}b')

def __binary_encoding(community_result: CD_RESULT_TYPE) -> pd.DataFrame:
    node_set = {node for community_list in community_result[list(CD_METHODS.keys())[0]] for node in community_list }
    df_features = pd.DataFrame(index=node_set)
    for meth, communities_dict in community_result.items():
        num_comms = len(communities_dict)
        node_comm_dict = {n:"0"*num_comms for n in node_set}
        for comm_idx, community in communities_dict.items():
            for node in community:
                node_comm_dict[node] = __int_to_binary(comm_idx, num_comms)
        df_features[meth] = df_features.index.map(node_comm_dict)
    return df_features

def get_graph_features(community_result: CD_RESULT_TYPE, encoding_method: ENCODING_TYPE = "binary") -> pd.DataFrame:
    if encoding_method not in ("binary", "one_hot"):
        raise ValueError(f"Invalid encoding_method: {encoding_method}. Must be 'binary' or 'one_hot'")
    
    if encoding_method == "binary":
        return __binary_encoding(community_result)
    # if `encoding_method` equals to `one_hot`
    return __one_hot_encoding(community_result)

def detect_communities(graph: nx.Graph) -> Dict[str, Dict[int, List[str | int]]]:
    community_result = {}
    for method_name, method_func in CD_METHODS.items():
        communities = method_func(graph)
        num_comms = len(communities.communities)
        if num_comms <= 1:
            warn(f"The <{method_name}> algorithm was excluded from further analysis as it identified only a single community.", UserWarning)    
            continue
        communities_dict = {}
        for idx, community in enumerate(communities.communities, start=1):
            communities_dict[idx] = community
        community_result[method_name] = communities_dict
    return community_result
        
def assign_by_neighbors(test_graph: nx.Graph, test_node: str|int, community_result: CD_RESULT_TYPE) -> dict:
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

def assign_by_modularity(test_graph: nx.Graph, test_node: str|int, community_result: CD_RESULT_TYPE) -> dict:
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

def get_test_node_community_features(
        test_node: str|int, 
        test_node_meth_comm: dict,
        community_result: CD_RESULT_TYPE, 
        encoding_method: ENCODING_TYPE = "binary"
    ) -> pd.DataFrame:

    if encoding_method not in ("binary", "one_hot"):
        raise ValueError(f"Invalid encoding_method: {encoding_method}. Must be 'binary' or 'one_hot'")
    
    if encoding_method == "binary":
        df = pd.DataFrame(index=[test_node], columns=list(test_node_meth_comm.keys()))
        for meth, comm_idx in test_node_meth_comm.items():
            binary_length = 5
            df.at[test_node, meth] = __int_to_binary(num=comm_idx, length=binary_length)
        return df
    # if `encoding_method` equals to `one_hot`
    df = __get_test_node_one_hot_features(
        test_node=test_node,
        test_node_meth_comm=test_node_meth_comm,
        community_result=community_result
    )
    return df
        

def __get_test_node_one_hot_features(test_node: str|int, test_node_meth_comm: dict, community_result: CD_RESULT_TYPE) -> pd.DataFrame:
    columns = []
    for meth, communities_dict in community_result.items():
        num_comm = len(communities_dict)
        for idx in range(1, num_comm + 1):
            columns.append(f"{meth}_comm_{idx}")
    df = pd.DataFrame(index=[test_node], columns=columns)
    for meth, communities_dict in community_result.items():
        num_comm = len(communities_dict)
        comm_idx = test_node_meth_comm[meth]
        for idx in range(1, num_comm + 1):
            col_name = f"{meth}_comm_{idx}"
            if idx == comm_idx:
                df.at[test_node, col_name] = 1
            else:
                df.at[test_node, col_name] = 0
    return df