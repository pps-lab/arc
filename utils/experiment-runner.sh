#!/bin/bash
# Expects the following arguments
# $1 ... Path to experiment code dir
# $2 ... player-id
# $3 ... seconds to sleep

export PYTHONPATH="$1/utils"
$1/.venv/bin/python -m python_utils.scripts.experiment_runner --player-number $2 --sleep-time $3