"""This module defines the code to process the raw text output of the MPC protocol virtual machines and move the output into
the results folder of the current experiment run for further processing.

The module defines the following functionalities:
- class OutputCapture:
    Implements the output processing to capture the raw textual output of the MPC protocol virtual machines and move the output into the results folder of the current experiment run for further processing.
"""
import os
import re
import sys


class OutputCapture:
    """Implements the output processing to capture the raw textual output of the MPC protocol virtual machines and move the output into the results folder of the current experiment run for further processing.

    Attributes
    ----------
    - output_prefix : str
        The prefix each raw output file will have
    - result_dir : str
        The path to the directory that contains the results directory, where all processed files need to be placed for further processing by the Design of Experiments suite
    - player_id : int
        The id of the current player
    - pattern : re.Patter (regular expression object)
        The compiled regular expression pattern that will be used to identify the raw textual output files that need to be processed.

    Methods
    -------
    - isrelevant(input_file_path, input_file):
        identifies if the input file found under the given input_file_path with name input_file is relevant for output processing
    - capture_output():
        captures the output of the MPC protocol virtual machine run and places the captured output into the results folder specified during the construction of the instance.
    """
    def __init__(self,output_prefix,result_dir,player_id):
        """
        Parameters
        ----------
        - output_prefix : str
            The prefix each raw output file will have
        - result_dir : str
            The path to the directory that contains the results directory, where all processed files need to be placed for further processing by the Design of Experiments suite
        - player_id : int
            The id of the current player
        """
        self.output_prefix = output_prefix
        self.result_dir = result_dir
        self.player_id = player_id
        self.pattern = re.compile(f"{output_prefix}-P([0-9]+)-([0-9]+)")

    def isrelevant(self, input_file_path, input_file):
        """identifies if the input file found under the given input_file_path with name input_file is relevant for output processing

        A file is judged to be relevant, if the input_file_path points to a file, and the name of the input file
        matches the regular expression pattern self.pattern.

        Parameters
        ----------
        - input_file_path : str
            absolute path to the input file to judge
        - input_file : str
            name of the input file to judge
        """
        if not(os.path.isfile(input_file_path)):
            return False

        match = self.pattern.match(input_file)
        if not(match):
            return False

        player_number = int(match.group(1))
        return player_number == self.player_id

    def capture_output(self):
        """captures the output of the MPC protocol virtual machine run and places the captured output into the results folder specified during the construction of the instance.

        The capture is done by first identifying all possible output files, and then copying the relevant output files into the
        results folder defined during the construction of the OutputCapture instance.
        """
        possible_input_files = os.listdir("./MP-SPDZ/")
        mp_spdz_path = os.path.join(os.getcwd(),"MP-SPDZ/")
        print(f"Capture_output: Possible input files: {possible_input_files}",file=sys.stderr)
        for input_file in possible_input_files:
            input_file_path = os.path.join(mp_spdz_path,input_file)
            if self.isrelevant(input_file_path,input_file):
                print(f"Captured_output: {input_file} is relevant", file=sys.stderr)
                match = self.pattern.match(input_file)
                player_id = int(match.group(1))
                thread_num = int(match.group(2))
                result_file_name = f"result-P{player_id}-{thread_num}.txt"
                result_file_path = os.path.join(self.result_dir,result_file_name)
                os.replace(input_file_path, result_file_path)
            else:
                # print(f"Captured_output: {input_file} is not relevant",file=sys.stderr)
                pass