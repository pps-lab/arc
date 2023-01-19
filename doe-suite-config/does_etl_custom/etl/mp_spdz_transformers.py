from multiprocessing.sharedctypes import Value
from doespy.etl.steps.extractors import Extractor
from doespy.etl.steps.transformers import Transformer
from doespy.etl.steps.loaders import Loader, PlotLoader
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt


def set_aggregate_f(s: pd.Series):
    return set(s)

def list_aggregate_f(s: pd.Series):
    return list(s)


SUPPORTED_AGGREGATE_FUNCTIONS = {
    "first": "first",
    "last": "last",
    "describe": "describe",
    "mean": "mean",
    "sum": "sum",
    "size": "size",
    "count": "count",
    "std": "std",
    "standard_deviation": "std",
    "var": "var",
    "variance": "var",
    "sem": "sem",
    "set_f": set_aggregate_f,
    "list_f": list_aggregate_f
}
DEFAULT_AGGREGATE_FUNCTION = SUPPORTED_AGGREGATE_FUNCTIONS['first']

class MpSpdzRowMergerTransformer(Transformer):
    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:
        df2 = df
        group_by_columns: list[str] = options['groupby_colums']
        
        def check_if_column_in_df(x):
            return x in df.columns    
        if not(all(map(check_if_column_in_df, group_by_columns))):
            # Fail fast and early
            return df

        df_groupby_obj = df.groupby(group_by_columns)

        # Now we construct the apply dictionary
        raw_apply_mapping: dict[str] = options['apply_columns']
        mapping_list = []
        for key in raw_apply_mapping.keys():
            new_column_name = key
            aggregate_column = raw_apply_mapping[key]['column_name']
            aggregate_function = SUPPORTED_AGGREGATE_FUNCTIONS.get(
                raw_apply_mapping[key]['func'], DEFAULT_AGGREGATE_FUNCTION
            )
            curr_named_agg = pd.NamedAgg(column=aggregate_column, aggfunc=aggregate_function)
            mapping_list.append((new_column_name,curr_named_agg))

        apply_mappings = dict(mapping_list)
        df2 = df_groupby_obj.agg(
            **apply_mappings
        )

        return df2


class MpSpdzDataFrameBuilderTransformer(Transformer):
    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:
        target_column_names = options['target_columns']
        # Check if target columes are contained
        def check_if_column_contained(x):
            return x in df.columns
        if not(all(map(check_if_column_contained, target_column_names))):
            print("Not all target columns exist. Aborting trasformation")
            return df

        rename_mapping = {
            k:  f"{k}_expanded" for k in target_column_names
        }
        selected_df = df[target_column_names]
        renamed_df = selected_df.rename(columns=rename_mapping)
        try:
            exploded_df = renamed_df.explode(list(renamed_df.columns)).dropna()
            return pd.concat([df, exploded_df])
        except ValueError as e:
            print(f"MpSpdzDataFameBuilderTransformer: {e}. Stopped transforming")
            return df 
