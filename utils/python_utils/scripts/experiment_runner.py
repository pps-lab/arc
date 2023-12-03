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
import sys

import python_utils.config_def as config_def
import python_utils.runner_defs as runner_defs
import python_utils.output_capture as out_cap
import python_utils.cleaner as clr
import python_utils.format_config as format_config
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
        custom_prime=task_config.custom_prime,
        custom_prime_length=task_config.custom_prime_length,
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

def prove_commitment_opening(task_config, output_prefix):
    if task_config.consistency_args is None:
        print("No consistency check arguments specified. Skipping consistency check.")
        return

    # GEN PP
    executable = f"target/release/gen_pp_{task_config.consistency_args.pc}"
    args = {
        "num-args": task_config.consistency_args.pp_args,
    }
    args_str = " ".join([f"--{k} {v}" for k,v in args.items()])
    executable_str = f"{executable} {args_str}"
    print(f"Generating public parameters with command: {executable_str}")

    result_dir_path = os.path.join(task_config.result_dir, DEFAULT_RESULT_FOLDER)
    consistency_gen_pp_output_file = open(os.path.join(result_dir_path, "consistency_gen_pp.log"), "w+")
    import subprocess
    subprocess.run(
        executable_str,
        shell=True,
        cwd=task_config.consistency_args.abs_path_to_code_dir,
        check=True,
        stdout=consistency_gen_pp_output_file,
        stderr=consistency_gen_pp_output_file,
    )

    # GEN COMMITMENTS
    executable = f"target/release/gen_commitments_{task_config.consistency_args.pc}"
    args = {
        "hosts": task_config.consistency_args.hosts_file,
        "party": task_config.player_id,
        "player-input-binary-path": f"{task_config.abs_path_to_code_dir}/MP-SPDZ/Player-Data/Input-Binary-P{task_config.player_id}-0",
        "save": "",
        "debug": "",
    }
    args_str = " ".join([f"--{k} {v}" for k,v in args.items()])
    executable_str = f"{executable} {args_str}"
    print(f"Running consistency check with command: {executable_str}")

    consistency_gen_commitments_output_file = open(os.path.join(result_dir_path, "consistency_gen_commitments.log"), "w+")
    import subprocess
    subprocess.run(
        executable_str,
        shell=True,
        cwd=task_config.consistency_args.abs_path_to_code_dir,
        check=True,
        stdout=consistency_gen_commitments_output_file,
        stderr=consistency_gen_commitments_output_file,
    )


    # mp_spdz_path = os.path.join(task_config.abs_path_to_code_dir, 'MP-SPDZ')
    # output_file = f"{output_prefix}-P{task_config.player_id}-0"
    result_file_name = f"consistency_poly_eval.log"
    result_file_path = os.path.join(result_dir_path, result_file_name)
    if not os.path.exists(result_file_path):
        print(f"Error: Could not find mpspdz output file! Expected to find {result_file_path}")
        return

    # PROVE_VERIFY
    executable = f"target/release/prove_verify_{task_config.consistency_args.pc}"
    args = {
        "hosts": task_config.consistency_args.hosts_file,
        "party": task_config.player_id,
        "mpspdz-output-file": result_file_path,
        "debug": "",
    }
    if task_config.consistency_args.prover_party is not None:
        args['prover-party'] = task_config.consistency_args.prover_party
    args_str = " ".join([f"--{k} {v}" for k,v in args.items()])
    executable_str = f"{executable} {args_str}"
    print(f"Running consistency check with command: {executable_str}")

    consistency_prove_verify_output_file = open(os.path.join(result_dir_path, "consistency_prove_verify.log"), "w+")
    import subprocess
    result_prove_verify = subprocess.run(
        executable_str,
        shell=True,
        cwd=task_config.consistency_args.abs_path_to_code_dir,
        check=True,
        stdout=consistency_prove_verify_output_file,
        stderr=consistency_prove_verify_output_file,
        # text=True
    )
    # print(result_prove_verify.stdout, file=sys.stdout)
    # print(result_prove_verify.stderr, file=sys.stderr)

def convert_shares(task_config):
    if task_config.commit_output is None and task_config.consistency_args is None:
        print("No commit or consistency check specified. No need to convert shares.")
        return

    protocol = task_config.protocol_setup

    conversion_not_needed = protocol == config_def.ProtocolChoices.REP_FIELD_PARTY or protocol == config_def.ProtocolChoices.SY_REP_FIELD_PARTY
    executable_prefix = None
    if protocol == config_def.ProtocolChoices.REPLICATED_RING_PARTY_X or protocol == config_def.ProtocolChoices.REP_FIELD_PARTY:
        executable_prefix = "rep"
    elif protocol ==  config_def.ProtocolChoices.SY_REP_RING_PARTY or protocol == config_def.ProtocolChoices.SY_REP_FIELD_PARTY:
        executable_prefix = "sy-rep"
    else:
        raise ValueError(f"Cannot convert from protocol {protocol}. Note that we can only convert from the ring for now.")
        # print("Cannot convert from protocol", protocol, ". Note that we can only convert from the ring for now.")
        # print("Continuing without converting shares.")

    player_data_dir = os.path.join(os.path.join(task_config.abs_path_to_code_dir, "MP-SPDZ"), "Player-Data")
    player_input_list, output_data, total_output_length = format_config.get_total_share_length(player_data_dir, task_config.player_count)

    total_input_length = 0
    player_input_counter = []

    spdz_args_str = f"-p {task_config.player_id} -N {task_config.player_count} -h {task_config.player_0_hostname}"

    if task_config.consistency_args is not None:

        if task_config.convert_ring_if_needed:
            executable = f"./{executable_prefix}-ring-switch-party.x"

            # shares is too slow because of the VM, we do input directly. In the future we should directly interface with MP-SPDZ!
            input_parts = []
            for player_id, p_inputs in enumerate(player_input_list):
                if p_inputs is None:
                    input_parts.append("-i 0")
                    player_input_counter.append(0)
                else:
                    types = []
                    player_input_cnt = 0
                    for p_input in p_inputs:
                        if p_input["type"] == "sfix":
                            types.append(f"f{p_input['length']}")
                        elif p_input["type"] == "sint":
                            types.append(f"i{p_input['length']}")
                        else:
                            raise ValueError(f"Unknown type {p_input['type']}")
                        total_input_length += p_input['length']
                        player_input_cnt += p_input['length']
                    player_input_counter.append(player_input_cnt)
                    input_parts.append(f"-i {','.join(types)}")
            input_str = " ".join(input_parts)

            executable_str = f"{executable} {spdz_args_str} --n_bits {task_config.convert_ring_bits} {input_str}"
            print(f"Converting input shares with command: {executable_str}")

            result_dir_path = os.path.join(task_config.result_dir, DEFAULT_RESULT_FOLDER)
            convert_shares_phase = open(os.path.join(result_dir_path, "consistency_convert_shares.log"), "a+")
            import subprocess
            try:
                subprocess.run(
                    executable_str,
                    shell=True,
                    cwd=os.path.join(task_config.abs_path_to_code_dir, "MP-SPDZ"),
                    check=True,
                    stdout=convert_shares_phase,
                    stderr=convert_shares_phase,
                )
            except subprocess.CalledProcessError as e:
                print("Error converting shares. Continuing without converting shares.")
                print(e)

        # compute a polynomial for each party
        input_counter = 0
        for party_id, player_input_count in enumerate(player_input_counter):
            if player_input_count == 0:
                print("Skipping party", party_id, "because it has no input.")
                continue
            # GEN PP
            executable = f"./{executable_prefix}-pe-party.x"
            args = {
                "n_shares": player_input_count, # convert all shares
                "start": input_counter,
                "input_party_i": party_id,
            }
            args_str = " ".join([f"--{k} {v}" for k,v in args.items()])
            executable_str = f"{executable} {spdz_args_str} {args_str}"
            print(f"Computing polynomial for player {party_id} with command: {executable_str}")

            result_dir_path = os.path.join(task_config.result_dir, DEFAULT_RESULT_FOLDER)
            poly_eval_phase = open(os.path.join(result_dir_path, "consistency_poly_eval.log"), "a+")
            import subprocess
            subprocess.run(
                executable_str,
                shell=True,
                cwd=os.path.join(task_config.abs_path_to_code_dir, "MP-SPDZ"),
                check=True,
                stdout=poly_eval_phase,
                stderr=poly_eval_phase,
            )
            input_counter += player_input_count
        assert input_counter == total_input_length, f"Expected to have processed {total_input_length} shares, but only processed {input_counter} shares."

    if task_config.commit_output is not None:

        if total_output_length == 0:
            print("No output to convert. Is this a mistake?")

        if task_config.convert_ring_if_needed:
            # convert the output shares
            executable = f"./{executable_prefix}-ring-switch-party.x"
            args = {
                "n_shares": total_output_length, # convert all shares
                # "start": None,
                "n_bits": task_config.convert_ring_bits,
                "out_start": total_input_length,
            }
            args_str = " ".join([f"--{k} {v}" for k,v in args.items()])
            executable_str = f"{executable} {spdz_args_str} {args_str}"
            print(f"Converting shares with command: {executable_str}")

            result_dir_path = os.path.join(task_config.result_dir, DEFAULT_RESULT_FOLDER)
            convert_shares_phase = open(os.path.join(result_dir_path, "consistency_convert_shares.log"), "a+")
            import subprocess
            try:
                subprocess.run(
                    executable_str,
                    shell=True,
                    cwd=os.path.join(task_config.abs_path_to_code_dir, "MP-SPDZ"),
                    check=True,
                    stdout=convert_shares_phase,
                    stderr=convert_shares_phase,
                )
            except subprocess.CalledProcessError as e:
                print("Error converting shares. Continuing without converting shares.")
                print(e)

        # check how many commitments we need
        # for each item in list output_data, add an arg with object_type
        args = { c['object_type']: c['length'] for c in output_data }
        args['s'] = total_input_length
        args_str = " ".join([f"-{k} {v}" for k,v in args.items()])

        executable = f"./{executable_prefix}-pc-party.x"

        executable_str = f"{executable} {spdz_args_str} {args_str}"
        print(f"Computing commitments with command: {executable_str}")

        result_dir_path = os.path.join(task_config.result_dir, DEFAULT_RESULT_FOLDER)
        poly_commit_phase = open(os.path.join(result_dir_path, "consistency_poly_commit.log"), "w+")
        import subprocess
        subprocess.run(
            executable_str,
            shell=True,
            cwd=os.path.join(task_config.abs_path_to_code_dir, "MP-SPDZ"),
            check=True,
            stdout=poly_commit_phase,
            stderr=poly_commit_phase,
        )


def clean_persistence_data(code_dir):
    """Clean the persistence data folder in the MP-SPDZ folder of the code workspace."""
    persistence_data_path = os.path.join(code_dir,"MP-SPDZ/Persistence")
    shutil.rmtree(persistence_data_path,ignore_errors=True)
    os.mkdir(persistence_data_path)


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
        clean_persistence_data(task_config.abs_path_to_code_dir)
        output_prefix=generate_random_prefix()
        run_script_with_args(task_config=task_config,
            output_prefix=output_prefix)
        capture_output(task_config=task_config,
            output_prefix=output_prefix)
        convert_shares(task_config=task_config)
        prove_commitment_opening(task_config=task_config,
                                 output_prefix=output_prefix)
        clean_workspace(task_config=task_config,output_prefix=output_prefix)


if __name__ == "__main__":
    cli()
