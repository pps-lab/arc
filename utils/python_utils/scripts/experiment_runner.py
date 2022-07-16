import python_utils.config_def as config_def
import python_utils.runner_defs as runner_defs
import python_utils.input_file_processor as ifp
import python_utils.output_capture as out_cap
import click
import os
import shutil
import string
import random

DEFAULT_CONFIG_NAME="config.json"
DEFAULT_RESULT_FOLDER="results/"

def generate_random_prefix() -> str :
    return ''.join([random.choice(string.ascii_lowercase + string.digits) for _ in range(20)])

def prepare_config(player_number,sleep_time):
    result_dir = os.getcwd()
    json_config_path = os.path.join(result_dir,DEFAULT_CONFIG_NAME) 
    json_config_obj = config_def.parse_json_config(config_path=json_config_path)
    task_config = config_def.build_task_config(
        json_cofig_obj=json_config_obj,
        player_number=player_number,
        sleep_time=sleep_time,
        result_dir=result_dir
    )
    return task_config

def move_to_experiment_dir(task_config: config_def.TaskConfig):
    os.chdir(task_config.abs_path_to_code_dir)

def process_input(task_config: config_def.TaskConfig):
    file_processor = ifp.InputFileProcessor(task_config.input_file_name)
    file_processor.process_input()


def copy_script_to_sources(task_config: config_def.TaskConfig):
    script_path = os.path.join(task_config.abs_path_to_code_dir, "scripts", f"{task_config.script_name}.mpc")
    source_path = os.path.join(task_config.abs_path_to_code_dir, "mp-spdz", "Programs", "Source", f"{task_config.script_name}.mpc")
    shutil.copy(script_path, source_path)

def compile_script_with_args(task_connfig: config_def.TaskConfig):
    copy_script_to_sources(task_config=task_connfig)
    comp_runner = runner_defs.CompilerRunner(
        script_name=task_connfig.script_name,
        script_args=task_connfig.script_args,
        compiler_args=runner_defs.CompilerArguments[task_connfig.protocol_setup.name]
    )
    comp_runner.run()

def run_script_with_args(task_config: config_def.TaskConfig, output_prefix: str):
    script_runner_constr: runner_defs.ScriptBaseRunner = runner_defs.ProtocolRunners[task_config.protocol_setup.name].value
    script_runner_obj = script_runner_constr(
        output_prefix=output_prefix,
        script_name=task_config.script_name,
        args=task_config.script_args,
        player_0_host=task_config.player_0_hostname,
        player_id=task_config.player_id
    )

def capture_output(task_config: config_def.TaskConfig, 
    output_prefix: str):
    result_dir_path = os.path.join(task_config.result_dir,DEFAULT_RESULT_FOLDER)
    out_cap_obj = out_cap.OutputCapture(output_prefix=output_prefix,
        result_dir=result_dir_path,
        player_id=task_config.player_id)
    out_cap_obj.capture_output()

    


@click.command()
@click.option("--player-number","player_number",required=True,
    type=float,
    help="The player number of the machine executing the experiment")
@click.option("--sleep-time","sleep_time",default=0.0,
    help="The amount of time the system should wait after compilation before proceeding to execution")
def cli(player_number,sleep_time):
    """This is the experiment runner script that will run the MP-SPDZ experiments for this framework"""
    task_config = prepare_config(
        player_number=player_number,
        sleep_time=sleep_time
    )
    move_to_experiment_dir(task_config=task_config)
    process_input(task_config=task_config)
    compile_script_with_args(task_connfig=task_config)
    output_prefix=generate_random_prefix()
    run_script_with_args(task_config=task_config,
        output_prefix=output_prefix)
    capture_output(task_config=task_config,
        output_prefix=output_prefix)
    
    
if __name__ == "__main__":
    cli()




    
