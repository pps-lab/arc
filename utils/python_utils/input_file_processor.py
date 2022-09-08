"""This module defines the code that processes the input file containers and places the contained input files into the correct folders,
namely into the Player-Data folder of the mp-spdz folder.

The module defines the following functionalities:
- class InputFileProcessor:
    Implements the input file processing logic for the Experiment Runner
"""
from ntpath import join
import zipfile
import os
import tempfile


DEFAULT_INPUT_FILE_PATH = "custom-data/"
DEFAULT_OUTPUT_FILE_PATH = os.path.join("mp-spdz","Player-Data")

class InputFileProcessor:
    """Implements the input file processing logic for the Experiment Runner
    
    Attributes
    ----------
    - input_file : str
        The name of the input file container to process
    
    Methods
    -------
    - process_input():
        Process the input file container defined during the construction of the object
    """
    def __init__(self, intput_file):
        """
        Parameters
        ----------
        - input_file : str
            The name of the input file container to process
        """
        self.input_file = intput_file
    
    def process_input(self):
        """Process the input file container defined during the construction of the object

        The processing is done as follows: First, all input files in the container are extractd into a Temporary Folder.
        Then, each input file in the temporary folder is moved to the Player-Data folder of the local MP-SPDZ installation.
        """
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
                