from ntpath import join
import zipfile
import os
import tempfile


DEFAULT_INPUT_FILE_PATH = "custom-data/"
DEFAULT_OUTPUT_FILE_PATH = os.path.join("mp-spdz","Player-Data")

class InputFileProcessor:
    def __init__(self, intput_file):
        self.input_file = intput_file
    
    def process_input(self):
        # Open Zip file
        zip_file_path = os.path.join(DEFAULT_INPUT_FILE_PATH,self.input_file)
        with zipfile.ZipFile(zip_file_path,"r") as input_zip:
            with tempfile.TemporaryDirectory() as tmp_dir:
                input_zip.extractall(path=tmp_dir)
                tmp_content = os.listdir(tmp_dir)
                for path in tmp_content:
                    file_path = os.path.join(tmp_dir,path)
                    if os.path.isfile(file_path):
                        # Move file to MP-SPDZ
                        dest_path = os.path.join(DEFAULT_OUTPUT_FILE_PATH, path)
                        os.replace(file_path,dest_path)
                