#!/bin/bash

export DOES_PROJECT_DIR="$(pwd)"
export PYTHONPATH="$(pwd)/utils:$(pwd)/script_utils:$(pwd)/mp-spdz/"
eval "$(ssh-agent -s)"

GITHUB_KEYS="<Please enter the path to your Github SSH Keys>"
AWS_KEYS="<Please enter the path to your AWS Keys  here>"

ssh-add "$GITHUB_KEYS"
ssh-add "$AWS_KEYS"

run_suite() {
    PREV_DIR="$(pwd)"
    cd "$DOES_PROJECT_DIR/doe-suite"
    poetry run ansible-playbook src/experiment-suite.yml -e "suite=$1 id=new $2"
    cd "$PREV_DIR"
}

continue_suite() {
    PREV_DIR="$(pwd)"
    cd "$DOES_PROJECT_DIR/doe-suite"
    poetry run ansible-playbook src/experiment-suite.yml -e "suite=$1 id=last $2"
    cd "$PREV_DIR"
}

clean_suite() {
    PREV_DIR="$(pwd)"
    cd "$DOES_PROJECT_DIR/doe-suite"
    poetry run ansible-playbook src/clear.yml
    cd "$PREV_DIR"
}

controller_connect() {
    PREV_DIR="$(pwd)"
    cd "$DOES_PROJECT_DIR/ansible-utils/setup-custom-controller"
    CONTROLLER_DOMAIN="$(poetry run ansible-inventory --list | jq '.alexmandt_ansible_controller.hosts[0]')"
    cd "$PREV_DIR"
    ssh controller@${CONTROLLER_DOMAIN:1:-1}
}

download_file() {
    PREV_DIR="$(pwd)"
    cd "$DOES_PROJECT_DIR/ansible-utils/setup-custom-controller"
    CONTROLLER_DOMAIN="$(poetry run ansible-inventory --list | jq '.alexmandt_ansible_controller.hosts[0]')"
    cd "$PREV_DIR"
    scp controller@${CONTROLLER_DOMAIN:1:-1}:~/$1 ./
}

controller_setup() {
    PREV_DIR="$(pwd)"
    cd "$DOES_PROJECT_DIR/ansible-utils/setup-custom-controller"
    poetry run ansible-playbook setup_controller.yml -e @tokens.yml
    cd "$PREV_DIR"
}

controller_cleanup() {
    PREV_DIR="$(pwd)"
    cd "$DOES_PROJECT_DIR/ansible-utils/setup-custom-controller"
    poetry run ansible-playbook destroy_aws.yml
    cd "$PREV_DIR"
}
