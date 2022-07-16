#!/bin/bash
# Expects the following arguments
# $1 ... Path to experiment code dir
# $2 ... player-id
# $3 ... seconds to sleep

export PYTHONPATH="$1/utils"
$1/.venv/bin/python "$1/utils/python_utils/scripts/experiment_runner.py" --player-number $2 --sleep-time $3