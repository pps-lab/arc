"""This module acts as the core implementation of the Experiment Runner. This modules is executed as a script and
executes an MPC experiment on a given experiment host and orchestrates the MPC experiment.

This module provides the following functionalities:

- function generate_random_prefix() : str
    Generates a random output prefix for use with the MPC protocol VMs to make the raw textual output of MPC protocol VM processes uniquely identifiable

- function prepare_config(player_number,sleep_time) : TaskConfigModel
    Exceutes the config model construction step.

- function move_to_experiment_dir(task_config : config_def.TaskConfig):
    Change the current working directory to the directory containing the code of the evaluation framework

- function compile_script_with_args(task_connfig: config_def.TaskConfig):
    Executes the Script compilation phase of the experiment.

- function run_script_with_args(task_config: config_def.TaskConfig, output_prefix: str):
    Executes the compiled script and configures the chosen MPC protocol VM to use output_prefix for its raw textual output

- function capture_output(task_config: config_def.TaskConfig, output_prefix: str):
    Captures the raw textual output using the OutputCapture class in the 'output_capture.py' module with output_prefix and moves the results to the results folders given in the task configuration.

- function clean_workspace(task_config: config_def.TaskConfig, output_prefix: str):
    Cleans the workspace from the output files with the prefix output_prefix and the data
    stored in the Player-Data and Player-Prep-Data folders.

- function cli(player_number,sleep_time):
    This is the entrypoint of the Experiment Runner

"""
import python_utils.config_def as config_def
import python_utils.runner_defs as runner_defs
import python_utils.output_capture as out_cap
import python_utils.cleaner as clr
import click
import os
import shutil
import string
import random

DEFAULT_CONFIG_NAME="config.json"
DEFAULT_RESULT_FOLDER="results/"

def generate_random_prefix() -> str :
    """Generates a random output prefix for use with the MPC protocol VMs to make the raw textual output of MPC protocol VM processes uniquely identifiable"""
    return ''.join([random.choice(string.ascii_lowercase + string.digits) for _ in range(20)])

def prepare_config(player_number,sleep_time):
    """Exceutes the config model construction step."""
    result_dir = os.getcwd()
    json_config_path = os.path.join(result_dir,DEFAULT_CONFIG_NAME)
    json_config_obj = config_def.parse_json_config(config_path=json_config_path)
    task_config = config_def.build_task_config(
        json_config_obj=json_config_obj,
        player_number=player_number,
        sleep_time=sleep_time,
        result_dir=result_dir
    )
    return task_config

def move_to_experiment_dir(task_config: config_def.TaskConfig):
    """Change the current working directory to the directory containing the code of the evaluation framework"""
    os.chdir(task_config.abs_path_to_code_dir)

def compile_script_with_args(task_config: config_def.TaskConfig):
    """Executes the Script compilation phase of the experiment."""
    compiler_args = task_config.compiler_args if task_config.compiler_args is not None else \
        runner_defs.CompilerArguments[task_config.protocol_setup.name].value
    comp_runner = runner_defs.CompilerRunner(
        script_name=task_config.script_name,
        script_args=task_config.script_args,
        compiler_args=compiler_args,
        code_dir=task_config.abs_path_to_code_dir
    )
    comp_runner.run()

def run_script_with_args(task_config: config_def.TaskConfig, output_prefix: str):
    """Executes the compiled script and configures the chosen MPC protocol VM to use output_prefix for its raw textual output"""
    script_runner_constr: runner_defs.ScriptBaseRunner = runner_defs.ProtocolRunners[task_config.protocol_setup.name].value
    script_runner_obj = script_runner_constr(
        output_prefix=output_prefix,
        script_name=task_config.script_name,
        args=task_config.script_args,
        player_0_host=task_config.player_0_hostname,
        player_id=task_config.player_id,
        player_count=task_config.player_count,
        program_args=task_config.program_args,
    )
    script_runner_obj.run()

def capture_output(task_config: config_def.TaskConfig,
    output_prefix: str):
    """Captures the raw textual output using the OutputCapture class in the 'output_capture.py' module with output_prefix and moves the results to the results folders given in the task configuration."""
    result_dir_path = os.path.join(task_config.result_dir,DEFAULT_RESULT_FOLDER)
    out_cap_obj = out_cap.OutputCapture(output_prefix=output_prefix,
        result_dir=result_dir_path,
        player_id=task_config.player_id)
    out_cap_obj.capture_output()

def clean_workspace(task_config: config_def.TaskConfig, output_prefix: str):
    """Cleans the workspace from the output files with the prefix output_prefix and the data
    stored in the Player-Data and Player-Prep-Data folders."""
    cleaner_obj = clr.Cleaner(code_dir=task_config.abs_path_to_code_dir,
        output_prefix=output_prefix)
    cleaner_obj.clean()

def run_consistency_check(task_config, output_prefix):
    if task_config.consistency_args is None:
        print("No consistency check arguments specified. Skipping consistency check.")
        return

    # GEN COMMITMENTS
    executable = f"target/release/gen_commitments_{task_config.consistency_args.pc}"
    args = {
        "hosts": task_config.consistency_args.hosts_file,
        "party": task_config.player_id,
        "player-input-binary-path": f"{task_config.abs_path_to_code_dir}/MP-SPDZ/Player-Data/Input-Binary-P{task_config.player_id}-0",
        "save": "",
    }
    args_str = " ".join([f"--{k} {v}" for k,v in args.items()])
    executable_str = f"{executable} {args_str}"
    print(f"Running consistency check with command: {executable_str}")

    import subprocess
    subprocess.run(
        executable_str,
        shell=True,
        cwd=task_config.consistency_args.abs_path_to_code_dir,
        check=True,
        capture_output=True
    )

    mp_spdz_path = os.path.join(task_config.abs_path_to_code_dir, 'MP-SPDZ')
    output_file = f"{output_prefix}-P{task_config.player_id}-0"
    if not os.path.exists(os.path.join(mp_spdz_path,output_file)):
        print(f"Error: Could not find mpspdz output file! Expected to find {output_file} in {mp_spdz_path}")
        return

    # PROVE_VERIFY
    executable = f"target/release/prove_verify_{task_config.consistency_args.pc}"
    args = {
        "hosts": task_config.consistency_args.hosts_file,
        "party": task_config.player_id,
        "mpspdz-output-file": os.path.join(mp_spdz_path, output_file),
    }
    args_str = " ".join([f"--{k} {v}" for k,v in args.items()])
    executable_str = f"{executable} {args_str}"
    print(f"Running consistency check with command: {executable_str}")

    import subprocess
    subprocess.run(
        executable_str,
        shell=True,
        cwd=task_config.consistency_args.abs_path_to_code_dir,
        check=True,
        capture_output=True
    )



# TODO: I think this could be simplified with e.g., a makefile?

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
    if "compile" in task_config.stage:
        compile_script_with_args(task_config=task_config)

    if "run" in task_config.stage:
        output_prefix=generate_random_prefix()
        run_script_with_args(task_config=task_config,
            output_prefix=output_prefix)
        capture_output(task_config=task_config,
            output_prefix=output_prefix)
        run_consistency_check(task_config=task_config,
                              output_prefix=output_prefix)
        clean_workspace(task_config=task_config,output_prefix=output_prefix)


if __name__ == "__main__":
    cli()
