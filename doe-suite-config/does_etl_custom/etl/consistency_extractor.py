

from doespy.etl.steps.transformers import Transformer
from doespy.etl.steps.extractors import Extractor


import pandas as pd
from typing import Dict, List
import re

class ConsistencyExtractor(Extractor):

    # transformer specific parameters with default values (see pydantic)
    # arg: str = None

    def file_regex_default(self):
        return [r"consistency_.*\.log$"]

    def extract(self, path: str, options: Dict) -> List[Dict]:
        print("MyExtractor: do nothing")

        with open(path, "r") as f:
            content = f.read()

        timer_data = self.extract_timers(content)
        stats_data = self.extract_stats(content)
        print(timer_data)
        print(stats_data)
        return timer_data + stats_data

    def extract_timers(self, content) -> List[Dict]:
        time_regex = r"^TIMER \(name=(.*)\) \(value=(\d*)\)$"
        time_matcher = re.compile(time_regex, re.MULTILINE)
        time_results = list(time_matcher.finditer(content))
        # go through all time_results and extract group 1 and 2
        # group 1 is the timer name
        # group 2 is the time value in microseconds
        time_results = list(map(lambda x: {
            "timer": self.remove_trailing_dots(x.group(1)), "timer_value": self.parse_time_to_millis(x.group(2))
        }, time_results))
        return time_results

    def parse_time_to_millis(self, time_str) -> float:
        return int(time_str) / 1000.0

    def extract_stats(self, content) -> List[Dict]:
        time_regex = r"^STAT \(name=(.*)\) \(value=(\d*)\)$"
        time_matcher = re.compile(time_regex, re.MULTILINE)
        time_results = list(time_matcher.finditer(content))

        time_results = list(map(lambda x: {
            "stat": self.remove_trailing_dots(x.group(1)), "stat_value": self.parse_time_to_millis(x.group(2))
        }, time_results))
        return time_results

    def extract_timers_flat(self, content) -> List[Dict]:
        time_regex = r"^.*End:\s*(.*)\.{3,}(\d.+)$"
        time_matcher = re.compile(time_regex, re.MULTILINE)
        time_results = list(time_matcher.finditer(content))
        # go through all time_results and extract group 1 and 2
        # group 1 is the timer name
        # group 2 is the time
        time_results = list(map(lambda x: {
            "timer": self.remove_trailing_dots(x.group(1)), "timer_value": self.parse_time_flat(x.group(2))
        }, time_results))
        return time_results

    def remove_trailing_dots(self, timer_str) -> str:
        # removes trailing dots from string
        # e.g. "Timer1..." -> "Timer1"
        return timer_str.rstrip(".").rstrip(" ")

    def parse_time_flat(self, time_str) -> float:
        # parses time_str and returns a float in ms
        # time_str and be in three different formats, depending on the suffix: s, ms and µs

        if time_str.endswith("ms"):
            time = float(time_str.strip()[:-2])
        elif time_str.endswith("µs"):
            time = float(time_str.strip()[:-2]) / 1000
        elif time_str.endswith("ns"):
            time = float(time_str.strip()[:-2]) / 1000000
        elif time_str.endswith("s"):
            time = float(time_str.strip()[:-1]) * 1000
        else:
            raise ValueError(f"Unknown time format: {time_str}")

        return time


