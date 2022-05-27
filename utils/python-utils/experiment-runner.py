# Experiment runner
# Replaces the old runner with this new runner

import subprocess
import argparse
import os
import json
import zipfile
import shutil
import time
from base64 import b64decode

class Base64DecodeAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, "json_config", b64decode(values))

class Base64Decoded2Action(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, "override_config", b64decode(values))

def do_preprocessing(config_obj=None, mpspdz_dir=None):
    if config_obj is None or mpspdz_dir is None:
        raise Exception("Missing parameters for do_preprocessing()")
    # Check if we can do preprocessing
    if 'preprocessing' in config_obj.keys() and \
        isinstance(config_obj['preprocessing'],list) :
        for task in config_obj.preprocessing:
            # Defining constants needed in each task
            data_path = os.path.join(mp_spdz_dir,'custom-data')
            player_data_path = os.path.join(mp_spdz_dir,'Player-Data')
            
            # Doing task switching
            if task['type'] == 'player_input':
                input_name = task['input_name']
                mappings = task['mappings']
                abs_input_path = os.path.join(data_path, f"{input_name}.zip")

                with zipfile.Zipfile(abs_input_path, 'r') as input_zip:
                    for i,name in enumerate(mappings):
                        # Extract element from zip file
                        curr_info = input_zip.getinfo(name)
                        extracted_elem_path = input_zip.extract(curr_info, path=data_path)
                        
                        # Move result to Player_Data folder with proper name
                        new_file_name = f"Input-P{i}-0"
                        new_file_path = os.path.join(player_data_path,new_file_name)
                        os.replace(extracted_elem_path, new_file_path)

            elif task['type'] == 'custom_cmd':
                # When a custom_cmd is specified, we just execute the command and leave the rest be
                cmd_string = task['command']
                subprocess.run(cmd_string, shell=True,cwd=data_path)

            elif task['type'] == 'custom_script':
                # We look for the file in custom-data and execute it
                script_file_name = task['script_fname']
                script_full_path = os.path.join(data_path,script_file_name)
                subprocess.run(["bash", script_full_path],cwd=data_path)
            else:
                # Other types of tasks are currently not known, so they are ignored
                print(f"experiment-runner.py: Unknown task type {task['type']} encountered for task {i}. Ignoring")
    else: # If task list is not given
        # Ignore empty task list
        pass 
    

def do_compilation(config_obj, mp_spdz_dir, experiment_dir):
    # We assume the script path is relative to the experiment dir
    # And we assume the full name of the script is given (with the .mpc ending)
    mpc_script_path = config['mpc_script_path']
    absolute_mpc_script_path = os.path.join(experiment_dir,mpc_script_path)

    # Prepare script for compilation
    mpc_script_basename = os.path.basename(absolute_mpc_script_path)
    new_mpc_script_name = f"custom-{mpc_script_basename}"
    new_mpc_script_absolute_path = os.path.join(mp_spdz_dir,"Programs","Source",new_mpc_script_name)
    shutil.copyfile(absolute_mpc_script_path,new_mpc_script_absolute_path)

    # Prepare for compilation
    arg_list = config_obj.get('args', [])
    if not(isinstance(arg_list,list)):
        raise Exception("experiment-runner.py: args given by config_object is not a list")
    

    # Do actual compilation
    # Decide for which setup we compile
    if config_obj['prot'] == "sh3":
        subprocess.run(["./compile.py", "-R", "64", "-C", "-D"] + [f"{new_mpc_script_name[:-4]}"] + arg_list + ["trunc_pr", "split3"],
            cwd=mp_spdz_dir)
    elif config_obj['prot'] == "mal_mascot":
        subprocess.run(["./compile.py", "-F", "64", "-C", "-D"] + [f"{new_mpc_script_name[:-4]}"] + arg_list,
            cwd=mp_spdz_dir)
    else:
        raise Exception(f"experiment-runner.py: Do not know protocol {config_obj['prot']}")
    return new_mpc_script_name


def run_compilation(config_obj, new_mpc_script_name,curr_dir, mp_spdz_dir):
    run_cmd = ""
    arg_list = config_obj.get('args',[])
    if config_obj['prot'] == "sh3":
        arg_part = "-".join(arg_list+["trunc_pr", "split3"])
    elif config_obj['prot'] == "mal_mascot":
        arg_part = "-".join(arg_list)
    else:
        raise Exception(f"experiment-runner.py: Do not know protocol {config_obj['prot']}")
    
    script_file_name = "-".join([new_mpc_script_name[:-4],arg_part])


    if config_obj['prot'] == "sh3":
        run_cmd = f"./replicated-ring-party.x -h {config_obj['player_0_dns']} -pn 12300 \"{config_obj['player_num']}\" {script_file_name} | tee {os.path.join(curr_dir,f'result-{player_num}')}" 
    elif config_obj['prot'] == "mal_mascot":
        run_cmd = f"./mascot-party.x -h {config_obj['player_0_dns']} -pn 12300 -p \"{config_obj['player_num']}\" {script_file_name} | tee {os.path.join(curr_dir,f'result-{player_num}')}" 
    
    subprocess.run(run_cmd, shell=True, cwd=mp_spdz_dir)

def do_post_processing(config_obj, mp_spdz_dir,new_mpc_script_name):
    # All Folders to delete
    data_prep_dir = os.path.join(mp_spdz_dir,"Player-Prep-Data")
    player_data_dir = os.path.join(mp_spdz_dir,"Player-Data")
    bytecode_dir = os.path.join(mp_spdz_dir,"Programs","Bytecode")
    publinc_input_dir = os.path.join(mp_spdz_dir,"Programs","Public-Input")
    schedule_dir = os.path.join(mp_spdz_dir,"Programs","Schedules")

    # File to delete
    mpc_script_path = os.path.join(mp_spdz_dir,"Programs","Source",new_mpc_script_name)

    # Delete folders
    shutil.rmtree(data_prep_dir)
    shutil.rmtree(player_data_dir)
    shutil.rmtree(bytecode_dir)
    shutil.rmtree(publinc_input_dir)
    shutil.rmtree(schedule_dir)

    # Recreate needed folders
    os.mkdir(data_prep_dir)
    os.mkdir(player_data_dir)

    #Delete file
    os.unlink(mpc_script_path)
    
    
    

def main():
    # Read arguments from stdin
    argument_parser = argparse.ArgumentParser()
    mutual_group = argument_parser.add_mutually_exclusive_group(required=True)
    mutual_group.add_argument("--b64-config", 
        help="JSON String providing the entire experiment configuration encoded as base64",
        action=Base64DecodeAction)
    mutual_group.add_argument("--json-config",
        help="JSON String providing the entire experiment configuration")
    mutual_group_2 = argument_parser.add_mutually_exclusive_group()
    mutual_group_2.add_argument("--override-config-b6",help="base64 encoded JSON string that overrides config values",
        action=Base64Decoded2Action)
    mutual_group_2.add_argument("--override-config-json", help="json string that overrides config values")

    args = argument_parser.parse_args()
    
    # Get config
    config = json.loads(args.json_config)
    # Override config
    override_config = json.loads(args.override_config)
    for sub in config:
        if sub in override_config:
            config[sub] = override_config[sub]

    curr_dir = os.getcwd()

    # Setup 
    experiment_dir = config.experiment_dir
    mp_spdz_dir = os.path.join(experiment_dir,"mp-spdz")

    # Preprocessing
    do_preprocessing(config_obj=config, mpspdz_dir=mp_spdz_dir)

    # Experiment Running part
    # We assume the script path is relative to experiment_dir
    new_mpc_script_name = do_compilation(config_obj=config, mp_spdz_dir=mp_spdz_dir,experiment_dir=experiment_dir)

    time.sleep(int(config['sleep_time']))

    run_compilation(config_obj=config, new_mpc_script_name=new_mpc_script_name,
        curr_dir=curr_dir, mp_spdz_dir=mp_spdz_dir)
    
    # Post-processing
    do_post_processing(config_obj=config, mp_spdz_dir=mp_spdz_dir,new_mpc_script_name=new_mpc_script_name)

    
    

    

if __name__ == "__main__":
    main()
