from os import makedirs
from os.path import join
import pandas as pd
from constants import Constants
from data_mgr import preprocess, get_fold_data
from learning import evaluate_classifiers, perform_pca, transform_test_pca
from tqdm import tqdm
from similarity_graph import generate_similarity_graph, add_test_node_to
from community_detection import (
    detect_communities, 
    get_graph_features,
    assign_by_neighbors,
    assign_by_modularity,
    get_test_node_community_features,
    CD_METHODS
)
from utils import concatenate_dataframes
import pickle

if __name__ == "__main__":

    my_directories = [
        Constants.OUT_DIR_PATH,
        Constants.GRAPH_DIR_PATH,
        Constants.FEATURE_DIR_PATH,
    ]

    for directory in my_directories:
        makedirs(directory, exist_ok=True)

    data = pd.read_csv(Constants.DATA_PATH)
    df_preprocessed = preprocess(dataframe=data)
    fold_data = get_fold_data(dataframe=df_preprocessed)
    evaluate_classifiers(fold_data=fold_data)

    for idx, (X_train, X_test, y_train, y_test) in tqdm(enumerate(fold_data, start=1), total=len(fold_data), desc="Processing folds"):
        graph_name = f"fold_{idx}_train"
        g_train = generate_similarity_graph(dataframe=X_train, graph_name=graph_name)
        # extract train feature 
        community_result = detect_communities(g_train)
        df_graph_feature = get_graph_features(community_result=community_result, encoding_method=Constants.ENCODING_METHOD)
        # pca
        if Constants.PAC_FLAG and Constants.ENCODING_METHOD == "one_hot":
            df_graph_feature, train_pca = perform_pca(graph_augmented_train_data=df_graph_feature)
        # concatenate features
        df_concatenated_features = concatenate_dataframes(X_train, df_graph_feature)
        # Save processed X_train
        df_concatenated_features.to_pickle(Constants.FEATURE_DIR_PATH + graph_name + '.pkl')
        # get test node and its feature from X_test
        # X_test only has one row
        test_node = X_test.index[0]
        test_feature = X_test.iloc[0]
        test_graph = add_test_node_to(
            train_graph=g_train,
            X_train=X_train,
            test_node=test_node,
            test_node_feat=test_feature
        )
        # heuristically add the test node to the train graph communities
        method_comm_dict = assign_by_modularity(
            test_graph=test_graph, test_node=test_node, community_result=community_result
        )

        df_test_feature = get_test_node_community_features(
            test_node=test_node,
            test_node_meth_comm=method_comm_dict,
            community_result=community_result,
            encoding_method=Constants.ENCODING_METHOD
        )
        if Constants.PAC_FLAG and Constants.ENCODING_METHOD == "one_hot":
            df_test_feature = transform_test_pca(graph_augmented_test_data=df_test_feature, fitted_pca=train_pca)
        # concatenate features
        df_concatenated_test_features = concatenate_dataframes(X_test, df_test_feature)
        # Save processed X_test
        df_concatenated_test_features.to_pickle(Constants.FEATURE_DIR_PATH + f"fold_{idx}_test.pkl")


    graph_augmented_fold_data = []
    for idx, (X_train, X_test, y_train, y_test) in tqdm(enumerate(fold_data, start=1), total=len(fold_data), desc="Preparing new data"):
        train_feat_path = join(Constants.FEATURE_DIR_PATH, f"fold_{idx}_train.pkl") 
        test_feat_path = join(Constants.FEATURE_DIR_PATH, f"fold_{idx}_test.pkl") 
        X_train_combined = pd.read_pickle(train_feat_path)
        X_test_combined = pd.read_pickle(test_feat_path)

        if Constants.ENCODING_METHOD == "binary":
            for meth in CD_METHODS.keys():
                X_train_combined[meth] = X_train_combined[meth].astype(int)
                X_test_combined[meth] = X_test_combined[meth].astype(int)

        graph_augmented_fold_data.append((X_train_combined, X_test_combined, y_train, y_test))

    with open(Constants.GRAPH_AUGMENTED_FOLD_DATA_PATH, 'wb') as f:
        pickle.dump(graph_augmented_fold_data, f)
    
    
        

    evaluate_classifiers(fold_data=graph_augmented_fold_data, primary=False)