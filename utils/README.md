# Cryptographic Auditing MPC: Utils

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
- `<abs-path-to-code-dir>` - Due to how doe-suite executes experiments, the script needs to know the absolute path of the code dir, so that it can find the MP-SPDZ dir, as MP-SPDZ expects to be executed at the place where all its binaries are located.
