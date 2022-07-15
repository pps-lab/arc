from ast import arg
import runner_defs
import click
import requests

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/996835329241788467/8BoaDmzY9SNeULgjJXgkX9e5qvrJz7aWrPuYk3V48XwSvYpm7EtjsP1aMu_Bd201GGwH"

def send_discord_message(message):
    request_params = {
        "content": message,
        "username": "LR Debug Messages",
    }
    requests.post(url=DISCORD_WEBHOOK_URL,json=request_params)



def run_learning_rate_experiment(lr,script_name,args):
    full_args = [lr]+args
    comp_run = runner_defs.CompilerRunner(script_name=script_name,
        script_args=full_args,
        compiler_args=runner_defs.CompilerArguments.EMULATE_X.value)
    emulator_runner = runner_defs.EmulatorRunner(
        output_prefix=f"lr-{lr}-results",
        script_name=script_name,
        args=full_args
    )

    send_discord_message(f"Staring Experiment for LR = {lr}")
    comp_run.run()
    send_discord_message(f"Compilation finished for LR = {lr}")
    emulator_runner.run()
    send_discord_message(f"Completed Experiment for LR = {lr}")

@click.command()
@click.argument("script-name",nargs=1)
@click.argument("args",nargs=-1)
def cli(script_name,args):
    """Finds the learning rate that works best for the unlearning"""
    send_discord_message("Start Learning Rate Experiment")
    learning_rates = [1e-2,5e-3,5e-4,5e-5,1e-5,5e-6,1e-6,5e-7,1e-7]
    n_lrs = len(learning_rates)
    for i,lr in enumerate(learning_rates):
        send_discord_message(f"{i} of {n_lrs}")
        run_learning_rate_experiment(lr,script_name,list(args))


if __name__ == "__main__":
    cli()
