from etl_base import Extractor, Loader, PlotLoader
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
import re

class MpSpdzStderrExtractor(Extractor):
    
    def file_regex_default(self):
        return ["^stderr\\.log$"]
    
    def extract(self, path: str, options: Dict) -> List[Dict]:
        with open(path, "r") as f:
            content = f.read()
        
        # Extract transfered data, round number and host
        matcher = re.compile(r'Data sent = ([0-9.]+) MB in ~([0-9]+) rounds \(party ([0-9]+)\)')
        results = matcher.search(content)
        if results:
            try:
                party_data_sent = float(results.group(1))
            except:
                party_data_sent = 0.0
            
            try:
                party_round_number = int(results.group(2))
            except:
                party_round_number = -1
            
            try:
                player_num = int(results.group(3))
            except:
                player_num = -1
            return [dict(player_num=player_num, player_data_sent=party_data_sent, player_round_number=party_round_number)]
        else:
            return []


class MpSpdzResultExtractor(Extractor):
    def file_regex_default(self):
        return ["^result-P[0-9]+-[0-9]\\.txt$"]
    
    def extract(self, path: str, options: Dict) -> List[Dict]:
        return []
        