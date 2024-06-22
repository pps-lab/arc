

from doespy.etl.steps.transformers import Transformer
from doespy.etl.steps.extractors import Extractor
from doespy.util import get_results_dir

import os
import pandas as pd
from typing import Dict, List
import re


class ConstantsExtractor(Extractor):

    path: str

    def default_file_regex():
        return [r"stdout\.log$"]

    def extract(self, path: str, options: Dict) -> List[Dict]:


        try:
            df = pd.read_csv(os.path.join(get_results_dir(), self.path))
        except:
            raise ValueError(f"Could not read csv file at {os.path.join(get_results_dir(), self.path)}."
                             f"Did you forget to generate `storage.csv` in `doe-suite-results`? Generate it using the notebook at `notebooks/storage.ipynb`.")

        # print(df)
        # print("DICT")
        # print(df.to_dict(orient="records"))
        # return df where each row is dict with column values
        return df.to_dict(orient="records")



class JoinWithCsvTransformer(Transformer):

    csv_path: str

    #join_columns: List[str] = None
    on: List[str] = None
    left_on: List[str] = None
    right_on: List[str] = None


    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        df1 = pd.read_csv(os.path.join(get_results_dir(), self.csv_path))

        df = df.merge(df1, on=self.on, left_on=self.left_on, right_on=self.right_on)

        return df