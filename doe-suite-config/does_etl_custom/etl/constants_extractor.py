

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


        df = pd.read_csv(os.path.join(get_results_dir(), self.path))

        print(df)
        print("DICT")
        print(df.to_dict(orient="records"))
        # return df where each row is dict with column values
        return df.to_dict(orient="records")
