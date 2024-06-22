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
import re
import sys
import warnings

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
import time

from python_utils import rendezvouz
from python_utils.consistency_cerebro import compile_cerebro_with_args, run_cerebro_with_args, compile_sha3_with_args, run_sha3_with_args

DEFAULT_CONFIG_NAME="config.json"
DEFAULT_RESULT_FOLDER="results/"

def generate_random_prefix() -> str :
    """Generates a random output prefix for use with the MPC protocol VMs to make the raw textual output of MPC protocol VM processes uniquely identifiable"""
    return ''.join([random.choice(string.ascii_lowercase + string.digits) for _ in range(20)])

def prepare_config(player_number):
    """Exceutes the config model construction step."""
    result_dir = os.getcwd()
    json_config_path = os.path.join(result_dir,DEFAULT_CONFIG_NAME)
    json_config_obj = config_def.parse_json_config(config_path=json_config_path)
    task_config = config_def.build_task_config(
        json_config_obj=json_config_obj,
        player_number=player_number,
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
                              output_prefix=output_prefix,
                              remove_input_files=task_config.remove_input_files)
    cleaner_obj.clean()

def sync_servers(task_config: config_def.TaskConfig):
    # ensure all servers are at this point
    # the way we do this is by pinging the host server which will respond when all are ready
    print("Syncing servers", flush=True)
    rendezvouz.sync(task_config.player_0_hostname, task_config.player_count, task_config.player_id)

def prove_commitment_opening(task_config, output_prefix):
    if task_config.consistency_args is None:
        print("No consistency check arguments specified. Skipping consistency check.")
        return
    if task_config.consistency_args.type != "pc":
        print(f"Consistency check type is {task_config.consistency_args.type}. Skipping PC phase.")
        return

    result_dir_path = os.path.join(task_config.result_dir, DEFAULT_RESULT_FOLDER)

    # GEN PP
    if task_config.consistency_args.gen_pp:
        executable = f"target/release/gen_pp_{task_config.consistency_args.pc}"
        args = {
            "num-args": task_config.consistency_args.pp_args,
        }
        args_str = " ".join([f"--{k} {v}" for k,v in args.items()])
        executable_str = f"{executable} {args_str}"
        print(f"Generating public parameters with command: {executable_str}")

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

    # time.sleep(2)
    # sync_servers(task_config)
    if task_config.sleep_time > 0:
        print(f"Sleeping for {task_config.sleep_time} seconds to allow gen commitment process on all clients to finish.")
        time.sleep(task_config.sleep_time)


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
    if task_config.sleep_time > 0:
        print(f"Sleeping for {task_config.sleep_time} seconds to allow commitment generation process on all clients to finish.")
        time.sleep(task_config.sleep_time)

    # mp_spdz_path = os.path.join(task_config.abs_path_to_code_dir, 'MP-SPDZ')
    # output_file = f"{output_prefix}-P{task_config.player_id}-0"
    result_file_name = f"consistency_poly_eval.log"
    result_file_path = os.path.join(result_dir_path, result_file_name)
    if not os.path.exists(result_file_path):
        print(f"Error: Could not find mpspdz output file! Expected to find {result_file_path}")
        return

    # sync_servers(task_config)
    if task_config.sleep_time > 0:
        print(f"Sleeping for {task_config.sleep_time} seconds to allow gen commitment process on all clients to finish.")
        time.sleep(task_config.sleep_time)

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

def cerebro_verify(task_config, input_size):
    executable = f"target/release/exponentiate_cerebro"
    args = {
        "n-parameters": input_size,
        "debug": "",
    }
    args_str = " ".join([f"--{k} {v}" for k,v in args.items()])
    executable_str = f"{executable} {args_str}"
    print(f"Running cerebro exponentiate with command: {executable_str}")

    result_dir_path = os.path.join(task_config.result_dir, DEFAULT_RESULT_FOLDER)
    consistency_cerebro_verify_output_file = open(os.path.join(result_dir_path, "consistency_cerebro_verify.log"), "a+")
    import subprocess
    result_cerebro_verify = subprocess.run(
        executable_str,
        shell=True,
        cwd=task_config.consistency_args.abs_path_to_code_dir,
        check=True,
        stdout=consistency_cerebro_verify_output_file,
        stderr=consistency_cerebro_verify_output_file,
        # text=True
    )

def convert_shares(task_config, output_prefix):
    if task_config.commit_output != True and task_config.consistency_args is None:
        print("No commit or consistency check specified. No need to convert shares.")
        return

    protocol = task_config.protocol_setup

    # If this is true, we will try to copy the share files (after conversion fails)
    conversion_not_needed = protocol == config_def.ProtocolChoices.REP_FIELD_PARTY or protocol == config_def.ProtocolChoices.SY_REP_FIELD_PARTY \
        or protocol == config_def.ProtocolChoices.LOWGEAR_PARTY or protocol == config_def.ProtocolChoices.HIGHGEAR_PARTY or protocol == config_def.ProtocolChoices.MASCOT_PARTY \
        or protocol == config_def.ProtocolChoices.SEMI_PARTY

    executable_prefix = None
    conversion_prefix = None
    need_input_sharing = False
    if protocol == config_def.ProtocolChoices.REPLICATED_RING_PARTY_X:
        executable_prefix = "rep"
        conversion_prefix = "rep-ring"
    elif protocol == config_def.ProtocolChoices.REP_FIELD_PARTY:
        executable_prefix = "rep"
        conversion_prefix = "rep-field" # TODO
    elif protocol ==  config_def.ProtocolChoices.SY_REP_RING_PARTY:
        executable_prefix = "sy-rep"
        conversion_prefix = "sy-rep-ring"
    elif protocol == config_def.ProtocolChoices.SY_REP_FIELD_PARTY:
        executable_prefix = "sy-rep"
        conversion_prefix = "sy-rep-field" # TODO
    elif protocol == config_def.ProtocolChoices.SEMI_PARTY:
        executable_prefix = "semi"
        conversion_prefix = "semi"
        need_input_sharing = task_config.custom_prime is not None # we compute in the custom prime field. we assume BLS377 for now
    elif protocol == config_def.ProtocolChoices.LOWGEAR_PARTY or protocol == config_def.ProtocolChoices.HIGHGEAR_PARTY or protocol == config_def.ProtocolChoices.MASCOT_PARTY:
        executable_prefix = "mascot"
        conversion_prefix = "mascot"
        need_input_sharing = task_config.custom_prime is not None # we compute in the custom prime field. we assume BLS377 for now
    else:
        raise ValueError(f"Cannot convert from protocol {protocol}.")

    player_data_dir = os.path.join(os.path.join(task_config.abs_path_to_code_dir, "MP-SPDZ"), "Player-Data")
    player_input_list, output_data, total_output_length = format_config.get_total_share_length(player_data_dir, task_config.player_count)

    debug_flag = "-d" if task_config.convert_debug else ""
    split_flag = "-sp" if task_config.consistency_args.use_split else ""

    total_input_length = 0
    # represents for each player id (key), a list of input checks that need to be run
    player_input_counter = { i: [] for i in range(len(player_input_list)) }

    spdz_args_str = f"-p {task_config.player_id} -N {task_config.player_count} -h {task_config.player_0_hostname}"

    # if task_config.sleep_time > 0:
    #     print(f"Sleeping for {task_config.sleep_time + task_config.post_spdz_sleep_time} seconds to allow the MP-SPDZ process on all clients to finish.")
    #     time.sleep(task_config.sleep_time + task_config.post_spdz_sleep_time)

    if task_config.consistency_args is not None:

        if (task_config.convert_ring_if_needed and
            task_config.consistency_args.type != "sha3" and task_config.consistency_args.type != "sha3s"): # manual check to avoid error
            executable = f"./{conversion_prefix}-switch-party.x"
            if need_input_sharing:
                print("Actually doing input sharing instead of conversion. We need to adapt this if we are also going to mascot convert")
                executable = f"./{executable_prefix}-share-party.x"

            # shares is too slow because of the VM, we do input directly. In the future we should directly interface with MP-SPDZ!
            # See also: https://github.com/data61/MP-SPDZ/issues/1257
            input_parts = []
            for player_id, p_inputs_objs in enumerate(player_input_list):
                if p_inputs_objs is None:
                    input_parts.append("-i 0")
                    player_input_counter[player_id].append(0)
                else:
                    types = []
                    for p_input_obj in p_inputs_objs:
                        p_inputs = p_input_obj["items"]
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
                        player_input_counter[player_id].append(player_input_cnt)
                    input_parts.append(f"-i {','.join(types)}")
            input_str = " ".join(input_parts)

            executable_str = f"{executable} {spdz_args_str} --n_bits {task_config.convert_ring_bits} --n_threads {task_config.convert_n_threads} --chunk_size {task_config.convert_chunk_size} {split_flag} {debug_flag} {input_str}"
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
                print(f"Error converting shares. Continuing without converting shares. {conversion_not_needed}")
                print(e)
                copy_transaction_files(conversion_not_needed, task_config)

            if task_config.sleep_time > 0:
                print(f"Sleeping for {task_config.sleep_time} seconds to allow the process on all clients to finish.")
                time.sleep(task_config.sleep_time)

        if task_config.consistency_args.type == "cerebro":
            # Re-invoke MP-SPDZ with script to compute the commitment
            # This can be grealy simplified once we integrate this functionality into MP-SPDZ
            print("Invoking cerebro to compute the commitments.")
            compile_cerebro_with_args(task_config, "single_cerebro") # standalone_cerebro
            run_cerebro_with_args(task_config, "single_cerebro", output_prefix, DEFAULT_RESULT_FOLDER, "input")

            # now we need to verify the commitment output
            for player_id, inputs in player_input_counter.items():
                for input_size in inputs:
                    if input_size > 0:
                        print(f"CEREBRO_INPUT_SIZE=({player_id},{input_size})", file=sys.stderr)
                        cerebro_verify(task_config, input_size)
        elif task_config.consistency_args.type == "cerebro_ec":
            # run poly_commit to compute cerebro exponentiation


            # now we need to verify the commitment output
            for player_id, inputs in player_input_counter.items():
                for input_size in inputs:
                    if input_size > 0:
                        print(f"CEREBRO_INPUT_SIZE=({player_id},{input_size})", file=sys.stderr)
                        cerebro_verify(task_config, input_size)
        elif task_config.consistency_args.type == "sha3":
            print("Computing sha3-based commitments, nothing else needed here.")
        elif task_config.consistency_args.type == "sha3s":
            print("Computing sha3-based commitments in separate script")
            compile_sha3_with_args(task_config, "standalone_sha3", True)
            run_sha3_with_args(task_config, "standalone_sha3", output_prefix, DEFAULT_RESULT_FOLDER, "input")
        else:
            # compute a polynomial for each party
            # log the data in player_input_counter
            print(f"Computing polynomials for the following inputs: {player_input_counter}")

            eval_point = None
            if task_config.consistency_args.eval_point is not None:
                eval_point = task_config.consistency_args.eval_point
            elif task_config.consistency_args.single_random_eval_point:
                print("Using the same eval point across poly eval runs."
                  "We will run the first eval script first and let the parties agree on a point."
                  "We then use this point in subsequent invocations.")
            else:
                warnings.warn("Note that if a party has multiple inputs to prove, we do not support using different points at the moment."
                              "It would be easy to support in the mpc-consistency code.")

            input_counter = 0
            for party_id, player_input_counts in player_input_counter.items():
                if len(player_input_counts) == 0:
                    print("Skipping party", party_id, "because it has no input.")
                    continue

                for player_input_count in player_input_counts:
                    if player_input_count == 0:
                        print("Skipping party", party_id, "because it has no input (player_input_count=0).")
                        continue
                    executable = f"./{executable_prefix}-pe-party.x"
                    args = {
                        "n_shares": player_input_count, # convert all shares
                        "start": input_counter,
                        "input_party_i": party_id,
                    }
                    if eval_point is not None:
                        args['eval_point'] = eval_point

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

                    # now look in log file for the eval point
                    if task_config.consistency_args.single_random_eval_point and eval_point is None:
                        print("Parsing eval point")
                        eval_point = find_eval_point(os.path.join(result_dir_path, "consistency_poly_eval.log"))

                    if task_config.sleep_time > 0:
                        print(f"Sleeping for {task_config.sleep_time} seconds to allow the process on all clients to finish.")
                        time.sleep(task_config.sleep_time)

            assert input_counter == total_input_length, f"Expected to have processed {total_input_length} shares, but only processed {input_counter} shares."

    if task_config.commit_output:

        if task_config.consistency_args.type == "sha3":
            print("Computing sha3 hash in script, nothing else needed here.")
        elif task_config.consistency_args.type == "sha3s":
            print("Computing sha3 hash in separate script")
            compile_sha3_with_args(task_config, "standalone_sha3", False)
            run_sha3_with_args(task_config, "standalone_sha3", output_prefix, DEFAULT_RESULT_FOLDER, "output")

            # TODO: Sign SHA as well?
        else:

            if total_output_length == 0:
                print("No output to convert. Is this a mistake?")

            if task_config.convert_ring_if_needed:
                # convert the output shares
                executable = f"./{conversion_prefix}-switch-party.x"
                args = {
                    "n_shares": total_output_length, # convert all shares
                    # "start": None,
                    "n_bits": task_config.convert_ring_bits,
                    "n_threads": task_config.convert_n_threads,
                    "chunk_size": task_config.convert_chunk_size,
                    "out_start": total_input_length,
                }
                args_str = " ".join([f"--{k} {v}" for k,v in args.items()])
                executable_str = f"{executable} {spdz_args_str} {split_flag} {debug_flag} {args_str}"
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
                    print(f"Error converting shares. Continuing without converting shares. {conversion_not_needed}")
                    print(e)
                    copy_transaction_files(conversion_not_needed, task_config)

                if task_config.sleep_time > 0:
                    print(f"Sleeping for {task_config.sleep_time} seconds to allow the process on all clients to finish.")
                    time.sleep(task_config.sleep_time)

            if task_config.consistency_args.type == "cerebro":
                # compute commitments
                print("Invoking cerebro to compute the commitments (output).")
                compile_cerebro_with_args(task_config, "single_cerebro")
                run_cerebro_with_args(task_config, "single_cerebro", output_prefix, DEFAULT_RESULT_FOLDER, "output")

                for c in output_data:
                    print(f"CEREBRO_OUTPUT_SIZE=({c['object_type']},{c['length']})", file=sys.stderr)
            else:
                pass

        # check how many commitments we need
        # for each item in list output_data, add an arg with object_type
        if task_config.consistency_args.type == "pc" or task_config.consistency_args.type == "cerebro_ec":
            print("Computing sha3-based commitments, nothing else needed here.")
            args = { c['object_type']: c['length'] for c in output_data }
            args['s'] = total_input_length
            args_str = " ".join([f"-{k} {v}" for k,v in args.items()])
        else:
            print("Will only sign commitments")
            args_str = ""

        executable = f"./{executable_prefix}-pc-party.x"

        executable_str = f"{executable} {spdz_args_str} {args_str}"
        print(f"Computing and signing commitments with command: {executable_str}")

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


def copy_transaction_files(conversion_not_needed, task_config):
    if conversion_not_needed:
        import re
        # need to add -P256 prefix to the persistence file
        persistence_data_path = os.path.join(task_config.abs_path_to_code_dir, "MP-SPDZ")
        persistence_data_path = os.path.join(persistence_data_path, "Persistence")
        print("copying persistence files in", persistence_data_path)
        # for each file in this dir
        for file_name in os.listdir(persistence_data_path):
            # if it is a file and in the format of regex Transactions-P(\d*)-0.data
            print("Found ", os.path.join(persistence_data_path, file_name))
            if os.path.isfile(os.path.join(persistence_data_path, file_name)) and re.match(r'Transactions-P(\d*)\.data',
                                                                                           file_name):
                # copy it to the persistence file with the prefix
                # add suffix before extension to filename
                filename_suffix = file_name.split(".")[0] + "-P251" + "." + file_name.split(".")[1]
                print("copying", file_name, "to", filename_suffix)
                shutil.copyfile(os.path.join(persistence_data_path, file_name),
                                os.path.join(persistence_data_path, filename_suffix))

def find_eval_point(filename):
    eval_point_regex = r"input_consistency_player_(\d*)_eval=\((.*),(.*)\)"
    with open(filename, "r") as f:
        for line in f.readlines():
            match = re.match(eval_point_regex, line)
            if match:
                print("Found eval point", match.group(2))
                return match.group(2)

    raise ValueError("Could not find eval point in log file.")


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
def cli(player_number):
    """This is the experiment runner script that will run the MP-SPDZ experiments for this framework"""
    task_config = prepare_config(
        player_number=player_number
    )
    move_to_experiment_dir(task_config=task_config)
    if "compile" in task_config.stage:
        clean_workspace(task_config=task_config, output_prefix=None)
        compile_script_with_args(task_config=task_config)

    if "run" in task_config.stage:
        clean_persistence_data(task_config.abs_path_to_code_dir)
        output_prefix=generate_random_prefix()
        run_script_with_args(task_config=task_config,
            output_prefix=output_prefix)
        sync_servers(task_config=task_config)
        convert_shares(task_config=task_config, output_prefix=output_prefix)
        prove_commitment_opening(task_config=task_config,
                                 output_prefix=output_prefix)
        capture_output(task_config=task_config,
                       output_prefix=output_prefix)


if __name__ == "__main__":
    cli()
