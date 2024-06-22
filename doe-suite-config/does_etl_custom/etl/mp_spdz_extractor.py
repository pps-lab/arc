import os.path

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

    ignore_errors = False

    verbose: bool = False

    def default_file_regex():
        return ["^stderr\\.log$"]

    def extract_cerebro_counts(self, content) -> List[Dict]:
        input_regex = "CEREBRO_INPUT_SIZE=\((\d*),(\d*)\)"
        input_results = re.findall(input_regex, content)

        n_input_cerebro = len(input_results)
        if n_input_cerebro > 0 and self.verbose:
            print("Found n_input results", n_input_cerebro)

        output_regex = "CEREBRO_OUTPUT_SIZE=\((.*),(\d*)\)"
        output_results = re.findall(output_regex, content)

        # add up second group
        n_output_cerebro = sum([int(x[1]) for x in output_results])
        if n_output_cerebro > 0 and self.verbose:
            print("Found n_output results", n_output_cerebro)

        output_list = []
        if n_input_cerebro > 0:
            output_list.append({"stat": f"n_input_cerebro", "stat_value": n_input_cerebro})
        if n_output_cerebro > 0:
            output_list.append({"stat": f"n_output_cerebro", "stat_value": n_output_cerebro})

        return output_list

    def extract(self, path: str, options: Dict) -> List[Dict]:
        with open(path, "r") as f:
            content = f.read()

        output_prefix = ""
        filename = os.path.basename(path)
        if filename != "stderr.log":
            output_prefix = "_".join(filename.split("_")[:-1]) + "_"

        error_regex = r"^Traceback \(most recent call last\):$"
        if not self.ignore_errors and re.search(error_regex, content, re.MULTILINE):
            raise Exception("Found error in file", path)

        # Extract transfered data, round number and host
        regex = r"Data sent = ([\d\.e\+]+) MB in ~([0-9]+) rounds"
        matcher = re.compile(regex)
        results = matcher.finditer(content)
        regex2 = r"Global data sent = ([\d\.e\+]+) MB \(all parties\)"
        matcher2 = re.compile(regex2)
        results2 = list(matcher2.finditer(content))
        time_regex = r"^Time([0-9]+)? = (.*) seconds \((.+) MB, (\d+) rounds\)$"
        time_matcher = re.compile(time_regex, re.MULTILINE)
        time_results = list(time_matcher.finditer(content)) 
        dicts = []
        # TODO: Improve this code quality, this should not have to be a loop

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

            def map_timer(x):
                timer_number = x.group(1)
                timer_value = float(x.group(2))
                timer_mb = float(x.group(3))
                timer_rounds = int(x.group(4))
                if timer_number is None:
                    timer_number = -1
                else:
                    timer_number = int(timer_number)
                # print("OH OH", (timer_number,timer_value,timer_mb,timer_rounds))
                return (timer_number,timer_value,timer_mb,timer_rounds)
            mapped_timer_results = list(map(map_timer, time_results))
            dicts += [{'stat': f"{output_prefix}spdz_timer_{t_num}", 'stat_value': t_val, 'player_number': player_num} for (t_num,t_val,_,_) in mapped_timer_results]
            dicts += [{'stat': f"{output_prefix}spdz_timer_bw_{t_num}", 'stat_value': t_val, 'player_number': player_num} for (t_num,_,t_val,_) in mapped_timer_results]
            dicts += [{'stat': f"{output_prefix}spdz_timer_rounds_{t_num}", 'stat_value': t_val, 'player_number': player_num} for (t_num,_,_,t_val) in mapped_timer_results]

            additional_values = { "player_number": player_num, "player_data_sent": party_data_sent, "player_round_number": party_round_number, "global_data_sent": global_data_sent}

            for label, value in additional_values.items():
                dicts += [{'stat': f"{output_prefix}spdz_{label}", 'stat_value': value, 'player_number': player_num}]

        lists = self.extract_cerebro_counts(content)
        dicts += lists

        # print(dicts)
        return dicts


class MpSpdzResultExtractor(Extractor):

    verbose: bool = False

    def default_file_regex():
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
                    if self.verbose:
                        print(f"Processing line {line}")
                    the_dict = self._process(path=path, options=options, line=match_result.group(1))
                else:
                    the_dict = None
                if the_dict is not None:
                    dicts.append(the_dict)
        
        # Finally, we combine all dicts to a super dicts
        final_dicts = {}
        if len(dicts) == 0:
            return []
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

