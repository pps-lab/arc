# This is a script that generates the input files for the toy-example experiment
# and is ment for local execution.


import tempfile
import os
import zipfile
import shutil
import click

def generate_player_input(zipfile_name='toy-example-input.zip'):
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        input_file_names = [f'Input-P{i}-0' for i in range(0,2,1)]

        # Generate the input files
        for input_file_name in input_file_names:
            with open(os.path.join(tmp_dir_name,input_file_name), 'w') as input_file:
                input_file.write('1 2 3 4')
            
        # generate a zip file out of these input files
        zipfile_name = 'toy-example-intput.zip'
        with zipfile.ZipFile(os.path.join(tmp_dir_name,zipfile_name),mode='w') as zip_obj:
            for input_name in input_file_names:
                zip_obj.write(os.path.join(tmp_dir_name,input_name), arcname=input_name)
        
        # Move generated zipfile to current directory
        cur_dir = os.getcwd()
        shutil.move(os.path.join(tmp_dir_name, zipfile_name), os.path.join(cur_dir,zipfile_name))

@click.command()
@click.option('-f','--filename', help='The name of the input file container that should be generated for the toy-example experiment', default='toy-example-intput.zip')
def cli(filename):
    generate_player_input(zipfile_name=filename)

if __name__ == '__main__':
    cli()
