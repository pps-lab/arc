from etl_base import Extractor, Loader, PlotLoader
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
import re
import pathlib

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
    
    def _process(self, path: str, options: Dict, line: str) -> Dict:
        pass

    def extract(self, path: str, options: Dict) -> List[Dict]:
        dicts = []
        with open(path, 'r') as the_file:
            for line in the_file:
                if line.startswith("###OUTPUT:") and line.endswith("###"):
                    the_dict = self._process(path=path, options=options, line=line)
                else:
                    the_dict = None
                if the_dict is not None:
                    dicts.append(the_dict)
        
        # Finally, we combine all dicts to a super dicts
        final_dicts = {}
        for curr_dict in dicts:
            final_dicts[curr_dict['name']] = curr_dict['value']

        file_path_obj = pathlib.PurePath(path)
        result_info_regex = re.compile('result-P([0-9]+)-([0-9]+)')
        result_match = result_info_regex.match(file_path_obj.name)
        player_num = result_match.group(1)
        thread_num = result_match.group(2)
        final_dicts['player_number'] = int(player_num)
        final_dicts['thread_number'] = int(thread_num)
        return [final_dicts]


        