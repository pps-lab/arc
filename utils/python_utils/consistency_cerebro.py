# This file only exists because we do not have arithmetic to arithmetic conversion in MP-SPDZ yet.
# After running the main script and the share conversion, we then invoke this function to compute the secret shares of the input.
from python_utils import runner_defs, config_def


def compile_cerebro_with_args(task_config: config_def.TaskConfig):
    """Executes the Script compilation phase of the experiment."""
    compiler_args = task_config.compiler_args if task_config.compiler_args is not None else \
        runner_defs.CompilerArguments[task_config.protocol_setup.name].value

    if '-F 251' not in compiler_args:
        compiler_args.append('-F 251') # ensure we compile for the right size

    if '-R 64' in compiler_args:
        # ugly manual override
        compiler_args.remove('-R 64')

    if '-F 128' in compiler_args:
        # ugly manual override
        compiler_args.remove('-F 128')

    comp_runner = runner_defs.CompilerRunner(
        script_name="standalone_cerebro",
        script_args=task_config.script_args,
        compiler_args=compiler_args,
        code_dir=task_config.abs_path_to_code_dir
    )
    comp_runner.run()

def run_cerebro_with_args(task_config: config_def.TaskConfig, output_prefix: str):
    """Executes the compiled script and configures the chosen MPC protocol VM to use output_prefix for its raw textual output"""
    script_runner_constr: runner_defs.ScriptBaseRunner = runner_defs.ProtocolRunners[task_config.protocol_setup.name].value
    script_runner_obj = script_runner_constr(
        output_prefix=output_prefix,
        script_name="standalone_cerebro",
        args=task_config.script_args,
        player_0_host=task_config.player_0_hostname,
        player_id=task_config.player_id,
        custom_prime=task_config.custom_prime,
        custom_prime_length="8444461749428370424248824938781546531375899335154063827935233455917409239041",
        player_count=task_config.player_count,
        program_args=task_config.program_args,
    )
    script_runner_obj.run()