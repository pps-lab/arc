# Experiment-Runner: JSON Config

The experiment-runner.py expects to run with a config.json, where all the configuration options are defined.
The experiment runner is expected to be run with the doe-suite.

## Config options

The configuration is epxected to be under `mpc`.

The following configuration options are supported:

```yaml
mpc:
    preprocessing:
        - type: input_file
          filename: 'input_file' # full name of zip file as how it is expected in custom-data folder
                                 # Content of zip file will be placed into Player-Data folder as is
        - type: custom_script
          scriptname: 'script_file' # Name of script file without .sh ending in custom-data folder
    protoc: 'semi-honest-3' # Supported protocols for now are semi-honest-3 and malicious-mascot
    script: 'my-script-name' # Here, we expect the name of the script without .mpc ending. For now, a flat structure is expected
    player0_dns: 'private-dns' # Here, we need to specify the name

    
    
```

playernumber is also required as a positional argument
