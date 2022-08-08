from etl_base import Extractor, Loader, PlotLoader, Transformer
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
        group_by_columns: list[str] = options['groupby_colums']
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

