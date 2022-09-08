<!-- # Cryptographic Auditing MPC: Utils

Here, we have the scripts that currently power the examples used to test if MP-SPDZ and doe-suite work together.

We have the following files:

1. `toy-example.sh` - Script powering the toy-example
2. `ml-stuff` - Folder containing the scripts and the shell script for the ml-example

## toy-example.sh - Usage
The `toy-example.sh` script expects the following arguments

```
toy-example.sh <Player-ID> <num-of-secs-to-sleep> <total-num-players> <hostname-of-player-0> <abs-path-to-code-dir>
```

- `<Player-ID>` - The 0-based id of the player involved in the MP-SPDZ run
- `<num-of-secs-to-sleep>` - The number of seconds to sleep to allow for the different player in the MP-SPDZ run to syncronize
- `<total-num-players>` - The total number of entities involved in the MP-SPDZ run
- `<hostname-of-player-0>` - The hostname of the server that runs player 0. Player 0 has a special role in MP-SPDZ as it directs all the communication between the different players
- `<abs-path-to-code-dir>` - Due to how doe-suite executes experiments, the script needs to know the absolute path of the code dir, so that it can find the MP-SPDZ dir, as MP-SPDZ expects to be executed at the place where all its binaries are located. -->

# Cryptographic Auditing MPC: Utils

This folder cotains the `python_utils` package and the `experiment-runner.sh` script.

The `python_utils` package contains all the Python code for the evaluation framework. Here, the Experiment Runner is defined.

The `experiment-runner.sh` script is a wrapper script for the `python_utils/scripts/experiment-runner.py` Experiment Runner script.
To execute this script, the `PYTHONPATH` environment variable needs to be set. This is done by this script. 
The script is executed as follows:
```
experiment-runner.sh <path-to-code-dir> <player-number> <sleep-time>
```
The `<path-to-code-dir>` argument is needed for the `experiment-runner.sh` script to properly configure the environment variable.
The `<player-number>` argument and the `<sleep-time>` arguments are passed to the `experiment-runner.py` script as the 
`--player-number` and `--sleep-time` option values respectively.
