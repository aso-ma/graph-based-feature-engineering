from typing import Final, Literal

class Constants:
    # Paths
    DATA_PATH: Final[str] = "./data/tabular_data.csv"
    OUT_DIR_PATH: Final[str] = "./output/"
    OUT_DIR_PATH: Final[str] = "./output/"
    GRAPH_DIR_PATH = OUT_DIR_PATH + 'graph/'
    FEATURE_DIR_PATH = OUT_DIR_PATH + 'features/'
    FOLD_DATA_PATH = OUT_DIR_PATH + 'fold_data.pkl'
    PRIMARY_EVAL_RESULT_PATH = OUT_DIR_PATH + "primary_eval_result.pkl"
    SECONDARY_EVAL_RESULT_PATH = OUT_DIR_PATH + "secondary_eval_result.pkl"
    GRAPH_AUGMENTED_FOLD_DATA_PATH = OUT_DIR_PATH + 'graph_augmented_fold_data.pkl'
    
    # Classification target column name
    TARGET_COL: Final[str] = "Target"

    # Edge Removal Threshold
    EDGE_RM_TH: float = 0.5
    
    # Use Principal Component Analysis (PCA) for dimensionality reduction 
    PAC_FLAG = False
    
    # ENCODING METHOD
    ENCODING_METHOD: Literal["binary", "one_hot"] = "binary"