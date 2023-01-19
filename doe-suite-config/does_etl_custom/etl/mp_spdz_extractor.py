from doespy.etl.steps.extractors import Extractor
from doespy.etl.steps.transformers import Transformer
from doespy.etl.steps.loaders import Loader, PlotLoader
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
import re
import pathlib
import json

class MpSpdzStderrExtractor(Extractor):
    
    def file_regex_default(self):
        return ["^stderr\\.log$"]
    
    def extract(self, path: str, options: Dict) -> List[Dict]:
        with open(path, "r") as f:
            content = f.read()
        
        # Extract transfered data, round number and host
        regex = r"Data sent = ([0-9]+\.[0-9]+|[0-9]+) MB in ~([0-9]+) rounds \(party ([0-9]+); rounds counted double due to multi-threading\)"
        matcher = re.compile(regex)
        results = matcher.finditer(content)
        regex2 = r"Global data sent = ([0-9]+\.[0-9]+|[0-9]+) MB (all parties)"
        matcher2 = re.compile(regex2)
        results2 = list(matcher.finditer(content))
        time_regex = r"^Time([0-9]+)? = ([0-9]+\.[0-9]+|[0-9]+) seconds.*$"
        time_matcher = re.compile(time_regex, re.MULTILINE)
        time_results = list(time_matcher.finditer(content)) 
        dicts = []
        for result in results:
            try:
                party_data_sent = float(result.group(1))
            except:
                party_data_sent = 0.0
            
            try:
                party_round_number = int(result.group(2))
            except:
                party_round_number = -1
            
            try:
                player_num = int(result.group(3))
            except:
                player_num = -1

            if len(results2) == 1:
                global_data_sent = float(results2[0].group(1))
            else:
                global_data_sent = -1
            
            if 'timers' in options.keys():
                print("Timers processing")
                def map_timer(x):
                    timer_number = x.group(1)
                    timer_value = float(x.group(2))
                    if timer_number is None:
                        timer_number = -1
                    else:
                        timer_number = int(timer_number)
                    return (timer_number,timer_value)
                mapped_timer_results = list(filter(lambda x: (x[0] in options['timers']), map(map_timer, time_results)))
                dicts += [{'timer_number': t_num, 'timer_value': t_val, 'player_number': player_num} for (t_num,t_val) in mapped_timer_results]
                print(dicts) 

            dicts += [dict(player_number=player_num, player_data_sent=party_data_sent, player_round_number=party_round_number, global_data_sent=global_data_sent)]
        return dicts


class MpSpdzResultExtractor(Extractor):
    def file_regex_default(self):
        return ["^result-P[0-9]+-[0-9]+\\.txt$"]
    
    def _process(self, path: str, options: Dict, line: str) -> Dict:
        raw_content = line
        json_decoder = json.JSONDecoder()
        json_dict: Dict = json_decoder.decode(raw_content)
        return json_dict


    def extract(self, path: str, options: Dict) -> List[Dict]:
        dicts = []
        pattern_matcher = re.compile("###OUTPUT:(.*)###")
        with open(path, 'r') as the_file:
            for line in the_file:
                match_result = pattern_matcher.match(line)
                if match_result:
                    print(f"Processing line {line}")
                    the_dict = self._process(path=path, options=options, line=match_result.group(1))
                else:
                    the_dict = None
                if the_dict is not None:
                    dicts.append(the_dict)
        
        # Finally, we combine all dicts to a super dicts
        final_dicts = {}
        for curr_dict in dicts:
            repeat_val = curr_dict['repeat']
            if not repeat_val:
                final_dicts[curr_dict['name']] = curr_dict['value']
            else:
                curr_value = final_dicts.get(curr_dict['name'], None)
                if curr_value is None:
                    final_dicts[curr_dict['name']] = [curr_dict['value']]
                else:
                    final_dicts[curr_dict['name']] = curr_value + [curr_dict['value']]
        file_path_obj = pathlib.PurePath(path)
        result_info_regex = re.compile('result-P([0-9]+)-([0-9]+)')
        result_match = result_info_regex.match(file_path_obj.name)
        player_num = result_match.group(1)
        thread_num = result_match.group(2)
        final_dicts['player_number'] = int(player_num)
        final_dicts['thread_number'] = int(thread_num)
        return [final_dicts]

