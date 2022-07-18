import shutil
import os
import re


class Cleaner:
    def __init__(self, code_dir, output_prefix):
        self.code_dir = code_dir
        self.output_prefix = output_prefix
        self.pattern = re.compile(f"{output_prefix}-P([0-9]+)-([0-9]+)")

    def is_relevant(self, file_name):
        if not os.path.isfile(os.path.join(self.code_dir,"mp-spdz/",file_name)):
            return False
        match = self.pattern.match(file_name)
        if match:
            return True
        else:
            return False

    def clean_output(self):
        input_file_list = os.listdir(os.path.join(self.code_dir,"mp-spdz/"))
        for input_file_name in [file_name for file_name in input_file_list if self.is_relevant(file_name)]:
            input_file_path = os.path.join(self.code_dir,"mp-spdz/",input_file_name)
            os.remove(input_file_path)
        
    def clean_player_pred_data(self):
        player_prep_data_path = os.path.join(self.code_dir,"mp-spdz/Player-Prep-Data/")
        shutil.rmtree(player_prep_data_path,ignore_errors=True)
        os.mkdir(player_prep_data_path)
    
    def clean_player_data(self):
        player_data_path = os.path.join(self.code_dir,"mp-spdz/Player-Data")
        shutil.rmtree(player_data_path,ignore_errors=True)
        os.mkdir(player_data_path)


    def clean(self):
        self.clean_output()
        self.clean_player_data()
        self.clean_player_pred_data()