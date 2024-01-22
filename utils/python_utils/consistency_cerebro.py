# This file only exists because we do not have arithmetic to arithmetic conversion in MP-SPDZ yet.
# After running the main script and the share conversion, we then invoke this function to compute the secret shares of the input.
from python_utils import runner_defs, config_def
import os

def compile_cerebro_with_args(task_config: config_def.TaskConfig, script_name: str):
    """Executes the Script compilation phase of the experiment."""
    compiler_args = task_config.compiler_args if task_config.compiler_args is not None else \
        runner_defs.CompilerArguments[task_config.protocol_setup.name].value

    if '-F 251' not in compiler_args:
        compiler_args.append('-F 251') # ensure we compile for the right size

    if '-R 64' in compiler_args:
        # ugly manual override
        compiler_args.remove('-R 64')

    if '-R' in compiler_args:
        idx = compiler_args.index('-R')
        compiler_args.remove('-R')
        compiler_args.remove(compiler_args[idx])

    if '-Z' in compiler_args:
        idx = compiler_args.index('-Z')
        compiler_args.remove('-Z')
        compiler_args.remove(compiler_args[idx])
        compiler_args.append('-Y') # use edabits instead of split

    if '-F 128' in compiler_args:
        # ugly manual override
        compiler_args.remove('-F 128')

    comp_runner = runner_defs.CompilerRunner(
        script_name=script_name,
        script_args=task_config.script_args,
        compiler_args=compiler_args,
        code_dir=task_config.abs_path_to_code_dir
    )
    comp_runner.run()

def map_protocol_to_field(task_config: config_def.TaskConfig):
    protocol = task_config.protocol_setup
    print("Protocol is: ", protocol)
    if protocol == config_def.ProtocolChoices.REPLICATED_RING_PARTY_X:
        return config_def.ProtocolChoices.REP_FIELD_PARTY
    elif protocol == config_def.ProtocolChoices.SY_REP_RING_PARTY:
        return config_def.ProtocolChoices.SY_REP_FIELD_PARTY

    return protocol

def run_cerebro_with_args(task_config: config_def.TaskConfig, script_name: str, output_prefix: str, results_folder: str, std_prefix: str):
    """Executes the compiled script and configures the chosen MPC protocol VM to use output_prefix for its raw textual output"""
    result_dir_path = os.path.join(task_config.result_dir, results_folder)
    cerebro_stdout = open(os.path.join(result_dir_path, f"cerebro_{std_prefix}_stdout.log"), "a+")
    cerebro_stderr = open(os.path.join(result_dir_path, f"cerebro_{std_prefix}_stderr.log"), "a+")

    program_args = task_config.program_args
    if script_name == "single_cerebro":
        # adjust edabits batch size
        if program_args is None:
            program_args = {}
        program_args['b'] = "2500"

    script_runner_constr: runner_defs.ScriptBaseRunner = runner_defs.ProtocolRunners[map_protocol_to_field(task_config).name].value
    script_runner_obj = script_runner_constr(
        output_prefix=output_prefix,
        script_name=script_name,
        args=task_config.script_args,
        player_0_host=task_config.player_0_hostname,
        player_id=task_config.player_id,
        custom_prime='8444461749428370424248824938781546531375899335154063827935233455917409239041',
        custom_prime_length=task_config.custom_prime_length,
        player_count=task_config.player_count,
        program_args=program_args,
    )
    script_runner_obj.run(cerebro_stdout, cerebro_stderr)


def compile_sha3_with_args(task_config: config_def.TaskConfig, script_name: str, compute_input: bool):
    """Executes the Script compilation phase of the experiment."""
    compiler_args = task_config.compiler_args if task_config.compiler_args is not None else \
        runner_defs.CompilerArguments[task_config.protocol_setup.name].value

    script_args = task_config.script_args
    script_args["compute_input"] = compute_input

    comp_runner = runner_defs.CompilerRunner(
        script_name=script_name,
        script_args=task_config.script_args,
        compiler_args=compiler_args,
        code_dir=task_config.abs_path_to_code_dir
    )
    comp_runner.run()

def run_sha3_with_args(task_config: config_def.TaskConfig, script_name: str, output_prefix: str, results_folder: str, std_prefix: str):
    """Executes the compiled script and configures the chosen MPC protocol VM to use output_prefix for its raw textual output"""
    result_dir_path = os.path.join(task_config.result_dir, results_folder)
    sha3_stdout = open(os.path.join(result_dir_path, f"sha3_{std_prefix}_stdout.log"), "a+")
    sha3_stderr = open(os.path.join(result_dir_path, f"sha3_{std_prefix}_stderr.log"), "a+")

    program_args = task_config.program_args

    script_runner_constr: runner_defs.ScriptBaseRunner = runner_defs.ProtocolRunners[task_config.protocol_setup.name].value
    script_runner_obj = script_runner_constr(
        output_prefix=output_prefix,
        script_name=script_name,
        args=task_config.script_args,
        player_0_host=task_config.player_0_hostname,
        player_id=task_config.player_id,
        custom_prime=task_config.custom_prime,
        custom_prime_length=task_config.custom_prime_length,
        player_count=task_config.player_count,
        program_args=program_args,
    )
    script_runner_obj.run(sha3_stdout, sha3_stderr)