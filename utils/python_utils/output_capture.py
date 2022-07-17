import os
import re
import sys


class OutputCapture:
    def __init__(self,output_prefix,result_dir,player_id):
        self.output_prefix = output_prefix
        self.result_dir = result_dir
        self.player_id = player_id
        self.pattern = re.compile(f"{output_prefix}-P([0-9]+)-([0-9]+)")
    
    def isrelevant(self, input_file_path, input_file):
        if not(os.path.isfile(input_file_path)):
            return False
        
        match = self.pattern.match(input_file)
        if not(match):
            return False
        
        player_number = int(match.group(1))
        return player_number == self.player_id 

    def capture_output(self):
        possible_input_files = os.listdir(".")
        print(f"Capture_output: Possible input files: {possible_input_files}",file=sys.stderr)
        for input_file in possible_input_files:
            input_file_path = os.path.join(os.getcwd(),input_file)
            if self.isrelevant(input_file_path,input_file):
                print(f"Captured_output: {input_file} is relevant", file=sys.stderr)
                match = self.pattern.match(input_file)
                player_id = int(match.group(1))
                thread_num = int(match.group(2))
                result_file_name = f"result-P{player_id}-{thread_num}.txt"
                result_file_path = os.path.join(self.result_dir,result_file_name)
                os.replace(input_file_path,result_file_path)
            else:
                print(f"Captured_output: {input_file} is not relevant",file=sys.stderr)
            


