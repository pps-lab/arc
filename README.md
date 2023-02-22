# Cryptographic Auditing MPC

> An evaluation framework to evaluate arbitrary auditing functions.

This repository contains the evaluation framework and the auditing function prototype
developed by Alexander Mandt as part of his Bachelor Thesis *Cryptographic Auditing in ML*.




After setting up MP-SPDZ (with emulator), we can use the following commands to compile and emulate:
```
make emulate

make emulate-debug
```



## Background

The evaluation framework builds upon [MP-SPDZ](https://github.com/data61/MP-SPDZ/tree/master) and the
[Design of Experiments Suite](https://github.com/nicolas-kuechler/doe-suite/tree/main) and a basic
unterstanding of both frameworks is recommended before using this evaluation framework.

Concretely, the evaluation framework builds upon MP-SPDZ as its Mulit-Party Computation runtime.
Auditing functions that will be evaluated by the evaluation framework must use the MP-SPDZ-specific
Python-DSL for developing auditing functions. Please see  [the MP-SPDZ documentation](https://mp-spdz.readthedocs.io/en/latest/)
for more information about the DSL.

Please note that MP-SPDZ and the DoE suite are integrated as submodules of this git repository and the
submodules `mp-spdz` and `doe-suite` define the version of MP-SPDZ and the DoE suite currently used by the evaluation framework
respectively.

Note: To install SHAP, might require LLVM@9 for llvmlite

## Requirements

To use the evaluation framework, the following requirements need to be met:

+ MP-SPDZ requirements must be met if local execution should be possible (I copy the requirements from MP-SPDZ's README for convenience. Please look at the requirements
of the currently used MP-SPDZ version if in doubt):
  + GCC 5 or later (tested with up to 11) or LLVM/clang 5 or later (tested with up to 12). We recommend clang because it performs better.
  + MPIR library, compiled with C++ support  (use flag `--enable-ccc` when running configure). You can use `make -j8 tldr` to install locally
  + libsodium library, tested against 1.0.18
  + OpenSSL, tested against 1.1.1
  + Boost.Asio with SSL support (`libboost-dev` on Ubuntu), tested against 1.71
  + Boost.Thread for BMR (`libboost-thread-dev` on Ubuntu), tested against 1.71
  + x86 or ARM 64-bit CPU (the latter tested with AWS Gravitron and Apple Silicon)
  + Python 3.5 or later
  + NTL library for homomorphic encryption (optional; tested with NTL 10.5)
  + If using macOS, Sierra or later
  + Windows/VirtualBox: see [this issue](https://github.com/data61/MP-SPDZ/issues/557) for a discussion
+ Python 3.9 must be installed locally
+ Git must be installed locally (v2.25 or later is recommended (as the framework was tested with this version); the submodule functionality is required)
+ Ensure that you can checkout Github repositories with SSH [(see instructions for Github)](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)
+ Ubuntu 20.04 as the OS is recommended, as the evaluation framework has been tested with this OS version. Later versions should work too.
  For other Operating Systems, MP-SPDZ dependencies might not be easily sourced, but the DoE suite and the evaluation framework itself
  should work, in principal.

## Getting Started

In this section, we will go through all steps to setup the evaluation framework for use. Please note that
after each step, ensure that any changes done to files belonging to the evaluation framework (all files not under the `mp-spdz` and `doe-suite` subfolders) are pushed to the repository, or else, the changes will not take effect.

2. Checkout this Github repository into a Folder of your liking: (For this section, we checkout into the `mpc-audit` folder)
   ```
   git clone --recursive git@github.com:pps-lab/cryptographic-auditing-mpc.git mpc-audit
   ```
   The `--recursive` option ensures that all submodules in the folder are checked out together with the evaluation framework.

3.  Go into the folder in which the repository has been checked out into
    ```
    cd mpc-audit


5. (Optional) Install MP-SPDZ locally to run scripts locally:
   1. Step into the `mp-spdz` folder
      ```
      cd mp-spdz
      ```


   2. We first ensure that all local dependecies are installed:

      For Ubuntu 20.04, executing the following command is enough to install all dependencies:
      ```
      sudo apt-get install automake build-essential git libboost-dev libboost-thread-dev libntl-dev libsodium-dev libssl-dev libtool m4 texinfo yasm python3
      ```

      Note that because we have Python3.9 installed, we can skip the `python3` command, since `python3` will install
      Python 3.8 by default.
   2. compilation and install the prerequisites:
      ```
      make -j8 tldr
      ```
   3. Compile the additonal MP-SPDZ protocol virtual machines (as described by the `README.md` in the current folder (the `mp-spdz` folder))
   4. Step out of the `mp-spdz`
      ```
      cd ..
      ```

6. Finish preparations for the evaluation framework:
   1. Install the dependencies of the evaluation framework
      ```
      poetry install
      poetry update # if poetry install fails
      ```




## Local development
We aim to keep the git submodule of MP-SPDZ up-to-date with the main branch.
This means that to execute programs locally in the emulator using MP-SPDZ,
you need to link the custom program files that exist in the top level repo to MP-SPDZ.

These commands assume you are in the root of the repo:
1. Link scripts: `ln -s  $(pwd)/scripts/audit_mnist.mpc $(pwd)/MP-SPDZ/Programs/Source/audit_mnist.mpc`
2. Link script_utils `ln -s $(pwd)/script_utils $(pwd)/MP-SPDZ/Compiler/script_utils`

Then `cd MP-SPDZ` and use MP-SPDZ as usual:
```bash
./compile.py -R 64 audit_mnist
./emulate.x audit_mnist
```
To install the emulator, run `make emulate.x`.
On MacOS, you may have to adapt the `CONFIG` to point to the correct location of openssl (when installed using Homebrew)
```bash
MY_CFLAGS += -I/usr/local/opt/openssl/include -I/opt/homebrew/opt/openssl/include -I/opt/homebrew/include
MY_LDLIBS += -L/usr/local/opt/openssl/lib -L/opt/homebrew/lib -L/opt/homebrew/opt/openssl/lib
MY_CFLAGS += -I/usr/local/opt/openssl/include -I/opt/homebrew/opt/openssl/include -I/opt/homebrew/include
MY_LDLIBS += -L/usr/local/opt/openssl/lib -L/opt/homebrew/lib -L/opt/homebrew/opt/openssl/lib
```

## Usage

The evaluation framework itself is in essence a pre-configured *Design of Experiments Suite* experiment repository that provides a pre-defined experiment flow that integrates MP-SPDZ.
This section will describe how to make use of this preconfigured experiment flow to ease the evaluation of cryptographic auditing function.

Before the evaluation framework can be used, please load the `setup-env.local.sh` script into your Terminal, so that all the necessary environment flags are set.
This is done as follows:
```
# From the evaluation framework root
source setup-env.local.sh
```
### Experiment Definition

To define an MPC experiment, we need to do the following steps:

1. Create a copy of the `does_config/designs/experiment-runner-template.yml` and rename the copy to a unique experiment name.
   For example, rename the copy to `toy-example.yml`.

2. Now, customize the template to your liking, the template contains comments that will explain all the settings.

3. Once you are done, you can start the MPC experiment as you would start an *Design of Experiments Suite* experiment suite.

### Additional Configuration steps

The evaluation framework provides two preconfiguration roles:

+ `ml-example-setup-1` - This role does the last-step configurations for the evaluation framework. This includes downloading all input data containers in the AWS Input file bucket and connecting the evaluation framework to the pre-compiled MP-SPDZ framework of the base image.
+ `setup-poetry` - This role install poetry on the experiment host and initalizes the evaluation framework.

If you want to add additional configuration steps, you can create a role as described in the *Design of Experiments Suite*'s documentation.
However, please ensure that any additional configuration steps are executed after `ml-example-setup-1` and `setup-poetry` roles.

### Writing MPC scripts

When using the evaluation framework, you want to evaluate your own function implementations.
To do this, please create a new script file with a file name ending in `.mpc` in the `scripts` folder.
The evaluation framework expects all MPC scripts to be stored there.

Also, the evaluation framework provides functionality to import additional code into the MPC script files.
To do this, please create a new Python module in the `script_utils` folder. Your own modules can then be found under the
`Compiler.script_utils` module as a submodule. Please note that MP-SPDZ uses a Python-based DSL to implement high-level programs.
Please consult the MP-SPDZ documentation to the available DSL functionality.

The evaluation framework also comes with some additional utility modules. Please look at the `README.md` in the `script_utils` folder
for documentation on these modules.

### Defining additional Host types

The evaluation framework provides two sets of default host types (as described in the Getting Started section).
If you want to generate additional host types, you can define additonal host types during the execution of the
`repotemplate.py` script, by specifying additional host types in during the server generation. However, please note
that you always need two host types with the same settings. This is because one host type must exclusively provide the server
that executes the main MPC protocol VM process. This is because MP-SPDZ's protocol VMs communicate via first establishing a connection
to Party 0 in the MPC protocol run and then doing a more direct communication. And due to how the *Design of Experiments Suite* stores
its data, it is not possible to name hosts directly. Therefore, we need at least one host type that contains only one server,
so that we can uniquely name the primary MPC protocol VM process server and query the server's DNS name. This name
must be given to all MPC protocol VM processes, so that they can connect to the main MPC protocol VM process.

Also, the servers should have at least 40 GB of space, as the base image requires at least 30GB and there is some additional free-space.


### Defining additional ETL steps

The experiment template file `does_config/designs/experiment-runner-template.yml` provides a sample ETL pipeline
that integrates the MP-SDPZ specific Extractor classes `MpSpdzStderrExtractor` and `MpSpdzResultExtractor` into a pipeline.
These extractors are essential to utilize the output processing facilities of the evaluation framework.
For more documentation on the MP-SPDZ-specific Extractors and Transformers, see the `README.md` in the `does_config/etl` folder.
[(The README.md)](./does_config/etl/README.md)



## Troubleshooting

In this section, we discuss common problems that may be encountered during the usage of the evaluation framework and how they can be
dealt with.

### Common Tips

+ Should the evaluation framework not work properly and the output of the evaluation framework makes no sense,
  consider turning off the pretty printing. To do this, please comment out the following setting:
  ```
  [defaults]
  INVENTORY = src/inventory
  roles_path = ${PWD}/src/roles:${DOES_PROJECT_DIR}/does_config/roles
  ANSIBLE_CALLBACKS_ENABLED = community.general.selective
  inventory_ignore_extensions = ~, .orig, .bak, .ini, .cfg, .retry, .pyc, .pyo, .j2

  ansible_ssh_common_args = '-o StrictHostKeyChecking=no -o ForwardAgent=yes'

  # TODO: activate / deactivate to only show the pretty log output
  stdout_callback = community.general.selective # <- Comment this setting out to disable pretty printing and receive the full output

  ```
  in the file `doe-suite/ansible.cfg`

### `poetry install` failed - lock file is not in sync with `pyproject.toml`

This error results from the *Design of Experiments Suite* having a `poetry.lock` file where the installed dependencies are not in sync with
the project dependencies declared in `pyproject.toml` in the *Design of Experiments Suite* root folder.
If you use Poetry v1.2 and onwards, you can execute the command `poetry lock` to resolve this issue. For earlier Poetry versions,
`poetry update` can also solve this problem.


### The evaluation framework tries to resolve internal AWS DNS names, and fails to do so & Experiment fails in the middle of the run because of DNS resolution failure of a `*.compute.internal` address

This happens because the *Design of Experiments Suite* might load an AWS server instance under 2 names, both under its publicly
reachable DNS name and its internal DNS name. However, the *Design of Experiments Suite* will not be able to reach an
instance through its internal DNS name. This can lead to a failure of an experiment run. This failure can be fixed
by adding the following entry to the `doe-suite/src/resources/inventory/aws_ec2.yml.j2` file:
```
---
plugin: aws_ec2
regions:
  - eu-central-1
filters:
{% if is_ansible_controller %} # for the ansible controller, we only filter for controllers but not projects
  tag:Name: ansible_controller
{% else %}
  tag:prj_id: {{ prj_id }}
{% if not prj_clear %}
  instance-state-name: ["running"]
  tag:suite: {{ suite }}
{% endif %}
{% endif %}

# keyed_groups may be used to create custom groups
#leading_separator: False
strict: False
hostnames:                # <- Adding these lines will force the AWS inventory plugin to only name instances via their public DNS names
  - dns-name              # <-
keyed_groups:
  - prefix: ""
    separator: ""
    key: tags.prj_id

  - prefix: ""
    separator: ""
    key: tags.suite

  - prefix: ""
    separator: ""
    key: tags.exp_name

  - prefix: ""
    separator: ""
    key: tags.host_type

  - prefix: "is_controller"
    separator: "_"
    key: tags.is_controller

  - prefix: "check_status"
    separator: "_"
    key: tags.check_status

  - prefix: ""
    separator: ""
    key: tags.Name

```
