import pandas as pd

def concatenate_dataframes(df_primary_feature: pd.DataFrame, df_graph_feature: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df_primary_feature, df_graph_feature], axis=1)

