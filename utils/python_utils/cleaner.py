"""This module defines the code that cleans the code workspace.

The Cleaner class implements this functionality.
"""
import shutil
import os
import re


class Cleaner:
    """This class implements the workspace cleaning procedure.

    Attributes
    ----------
    - code_dir: str
        The path to the directory that contains the evaluation framework code.
    - output_prefix: str
        The output prefix of the text output files generated during the experiment run.
    - pattern: re.Pattern (regular expression object)
        The compiled regular expresion pattern used to match experiment output files.

    Methods
    -------
    - is_relevant(file_name): bool
        Check if the given file_name is the name of a valid file and is also the name of an experiment output file
    - clean_output():
        Clean the experiment output files
    - clean_player_prep_data():
        Clean the Player-Prep-Data folder in the MP-SPDZ folder of the code workspace.
    - clean_player_data():
        Clean the Player-Data folder in the MP-SPDZ folder of the code workspace.
    - clean():
        Cleans the code workspace.
    """
    def __init__(self, code_dir, output_prefix, remove_input_files):
        """
        Parameters
        ----------
        - code_dir : str
            The path to the directory that contains the evaluation framework code.
        - output_prefix : str
            The output prefix of the text output files generated during the experiment run.
        """
        self.code_dir = code_dir
        self.output_prefix = output_prefix
        self.pattern = re.compile(f"{output_prefix}-P([0-9]+)-([0-9]+)")
        self.remove_input_files = remove_input_files

    def is_relevant(self, file_name):
        """Check if the given file_name is the name of a valid file and is also the name of an experiment output file

        Parameters
        ----------
        - file_name : str
            Name of the file that should be checked.
        """
        if not os.path.isfile(os.path.join(self.code_dir,"MP-SPDZ/",file_name)):
            return False
        match = self.pattern.match(file_name)
        if match:
            return True
        else:
            return False

    def clean_output(self):
        """Clean the experiment output files"""
        input_file_list = os.listdir(os.path.join(self.code_dir,"MP-SPDZ/"))
        for input_file_name in [file_name for file_name in input_file_list if self.is_relevant(file_name)]:
            input_file_path = os.path.join(self.code_dir,"MP-SPDZ/",input_file_name)
            os.remove(input_file_path)

    def clean_player_pred_data(self):
        """Clean the Player-Prep-Data folder in the MP-SPDZ folder of the code workspace."""
        player_prep_data_path = os.path.join(self.code_dir,"MP-SPDZ/Player-Prep-Data/")
        shutil.rmtree(player_prep_data_path,ignore_errors=True)
        os.mkdir(player_prep_data_path)

    def clean_player_data(self):
        """Clean the Player-Data folder in the MP-SPDZ folder of the code workspace."""
        player_data_path = os.path.join(self.code_dir, "MP-SPDZ/Player-Data")
        for file_name in os.listdir(player_data_path):
            # if it is a file and in the format of regex Transactions-P(\d*)-0.data
            if os.path.isfile(os.path.join(player_data_path, file_name)) and re.match(r'Input-Binary-P(.*)',
                                                                                           file_name):
                print("Removing ", os.path.join(player_data_path, file_name))
                os.remove(os.path.join(player_data_path, file_name))
            if os.path.isfile(os.path.join(player_data_path, file_name)) and re.match(r'^Output-format$',
                                                                                      file_name):
                print("Removing ", os.path.join(player_data_path, file_name))
                os.remove(os.path.join(player_data_path, file_name))
        # shutil.rmtree(player_data_path,ignore_errors=True)
        # os.mkdir(player_data_path)

    def clean(self):
        """Cleans the code workspace."""
        self.clean_output()
        if self.remove_input_files:
            self.clean_player_data()
        # self.clean_player_pred_data()