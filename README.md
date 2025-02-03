
# Artifact Appendix
<a name="readme-top"></a>

This repository contains the artifact for the USENIX 2024 submission ["Holding Secrets Accountable: Auditing Privacy-Preserving Machine Learning"](https://arxiv.org/abs/2402.15780).

## Abstract
In this work, we introduce Arc, an MPC framework designed
for auditing privacy-preserving machine learning. 
Arc cryptographically ties together MPC training, inference, and auditing
phases to allow robust and private auditing. 
At the core of our framework is a new protocol for efficiently verifying inputs
against succinct commitments, ensuring the integrity of the training data, model and prediction samples across phases. 
We evaluate the performance
of our framework when instantiated with our consistency protocol and compare it to hashing-based and homomorphic commitment-based approaches, demonstrating that it is up to
10^4× faster and up to 10^6× more concise.

## Description & Requirements
The artifact is a prototype implementation of the Arc framework.
The prototype is designed to focus on evaluating the overheads of input consistency protocols for MPC computations.
The framework uses the MPC implementation of MP-SPDZ, a research framework execute MPC programs with different protocols. 
Arc extends the MPC protocols in MP-SPDZ with the ability to check that MPC inputs are consistent with a commitment through a novel consistency check protocol.

This repository is structured as follows.

### Arc Components
- [MPC auditing functions](scripts) 
Implementations of several auditing functions (in addition to ML training and inference) in [MP-SPDZ's](MP-SPDZ) DSL.

- [MPC consistency utils](script_utils)
A helper library with 1) functionality to load the correct datasets and models for auditing and 2) to compute the correct metadata for the inputs and outputs to run the consistency check protocol on.

- [Consistency check protocol](mpc-consistency) The code for the consistency check protocol based on polynomial commitments.
This component uses Arkworks' poly_commit library and is implemented in Rust.

- [Share conversion and efficient EC-MPC scripts](https://github.com/pps-lab/MP-SPDZ/tree/85c3c7e65bb7759c96ef540cca96f6a7163d3568) 
Our framework adds functionality to MP-SPDZ for share conversion and efficient EC-MPC operations, which
are implemented as lower-level MP-SPDZ scripts.

### Arc Benchmarks & Evaluation
- [DoE-Suite](https://github.com/nicolas-kuechler/doe-suite) The evaluation is built using the [DoE-Suite](https://github.com/nicolas-kuechler/doe-suite), which allows
straightforward reproducibility of results by defining experiments in a configuration file (_suite_) and executing them on a set of machines.
We provide the necessary DoE-Suite commands to reproduce all results. 
However, it is also possible to obtain the individual commands used to invoke the framework and run them manually.
- [Utils](utils/python_utils) [Experiment runner](utils/python_utils/scripts/experiment_runner.py) utility that ties together the MPC computation, the share conversion and the consistency protocols.
This script is the entrypoint for the remote servers when running the experiment and reads the experiment instance config file
created by doe-suite for each experiment.

  
### Security, privacy, and ethical concerns
There are no concerns when running this artifact locally.
Please note that executing experiments on your AWS infrastructure involves the creation of multiple EC2 instances, resulting in associated costs.
Please manually check that any created machines are terminated afterward.


### How to access
The artifact can be accessed at [https://github.com/pps-lab/arc/tree/ae_1](https://github.com/pps-lab/arc/tree/ae_1)

### Hardware dependencies
None

### Software dependencies
No private software is required for this artifact.

This artifact has been tested on Ubuntu and MacOS.
The artifact relies on DoE-Suite to install all necessary dependencies on the end-to-end servers.
To run DoE-Suite, we require Python 3.9, Poetry, AWS CLI and Make to be installed, which are also highlighted in the installation instructions below. 
The framework uses Poetry to manage further Python dependencies. 
The sub-components require additional dependencies, which must be installed manually if you wish to run these components locally (without DoE-Suite). 
In particular, [mpc-consistency](mpc-consistency) requires Rust to be installed.

### Benchmarks
No proprietary benchmarks and public datasets are automatically loaded by the build scripts.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Set-up
We provide a `make` command to run a [JupyterLab environment](artifact.ipynb) to run the experiments and evaluate the artifact.
This should ensure that the environment is set up correctly and that the necessary dependencies are installed.

### Installation
We require Python, Poetry and Make to be installed to run the artifact.
To get a local copy up and running follow these steps.

1. Local clone of the repository (**with submodules!**)
    ```sh
    git clone --recurse-submodules git@github.com:pps-lab/arc.git
    ```
   
2. Install [Python Poetry](https://python-poetry.org/)
    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```

3. Install [Make](https://www.gnu.org/software/make/)

4. Install [Install AWS CLI version 2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) (to run remote experiments on AWS)

#### Environment Variables
The [doe-suite](https://github.com/nicolas-kuechler/doe-suite/) requires a few environment variables and should handle
the rest of the configuration automatically (including for the Jupyter notebook) using relative paths and poetry.

Setup environment variables for the Doe-Suite:
 ```sh
 # root project directory (expects the doe-suite-config dir in this folder)
 export DOES_PROJECT_DIR=<PATH>

 #  Your unique short name, such as your organization's acronym or your initials.
 export DOES_PROJECT_ID_SUFFIX=ae
 ```

For AWS EC2:
 ```sh
 export DOES_CLOUD=aws

 # name of ssh key used for setting up access to aws machines (name of key not path)
 export DOES_SSH_KEY_NAME=id_ec_arc
 ```
 `DOES_SSH_KEY_NAME` refers to the key the reviewers have received through artifact evaluation system, for more info see below.

> Tip: To make it easier to manage project-specific environment variables, you can use a tool like [Direnv](https://direnv.net/docs/installation.html). Direnv allows you to create project-specific .envrc files that set environment variables when you enter the project directory.
With Direnv, the below environment variables would be set in [doe-suite/.envrc](doe-suite/.envrc)


#### Setting up AWS
Authentication details can be found in the Artifact submission system. This will allow the Artifact reviewers to run the evaluation on the same resources stated in the paper submission. The experiments on AWS are automated with DoE-Suite and can be called from the JupyterLab environment.

Reviewers should have received a private key: `id_ec_arc` and AWS credentials
1. Move the provided private key `id_ec_arc` to the `.ssh` folder of your home directory (reviewers should have received the key, otherwise contact us).
   Ensure the id_ec_arc key has the correct permissions:
   ```shell
   chmod 600 ~/.ssh/id_ec_arc
   ```

2. Configure AWS credentials using `aws configure`.
   The credentials can be found in the `arc_ae_accessKeys.txt`.
   Set `eu-central-1` as the default region.
   By default, credentials should be stored in ``~/.aws/credentials``.

3. To configure SSH access to AWS EC2 instances, you need to add a section to your ``~/.ssh/config`` file:
   ```
   Host ec2*
      IdentityFile ~/.ssh/id_ec_arc
      User ubuntu
      ForwardAgent yes
      StrictHostKeyChecking no
      UserKnownHostsFile=/dev/null
   ```

For more details on the installation of doe-suite please refer to the [doe-suite documentation](https://nicolas-kuechler.github.io/doe-suite/installation.html#base-installation)
and [AWS-specific instructions](https://nicolas-kuechler.github.io/doe-suite/installation.html#aws-specific).

#### Running JupyterLab
From this point on it is possible to run the Jupyter notebook which contains the experiments and evaluation of the artifact.
To launch the Jupyter notebook in the environment with the correct dependencies, we provide a make command.
1. Navigate to the `doe-suite` sub-directory and run `make jupyter` which will launch these instructions in the form of a notebook.
```shell
cd doe-suite
make jupyter
```
2. Run the below cell to check that it prints `Environment loaded successfully`.
<!-- Cell with env -->


If you see any errors, make sure the correct environment variables are set.

Our JupyterLab environment contains further documentation, runnable code cells to run benchmarks locally or on AWS (using `!` to run shell commands), and runnable code cells to recreate the plots from the paper.

To execute a selected cell, either click on the run button in the toolbar or use `Ctrl + Enter`. To keep the JupyterLab readable, cells containing longer code sections to create plots are initially collapsed (indicated by three dots). However, by clicking on the dots, the cell containing the code expands (`View/Collapse Selected Code` to collapse the code again).

**Please ensure that the AWS resources are cleaned up if they are not used in the artifact**
If the playbook terminates normally (i.e., without error and is not interrupted), then the playbook ensures that the created resources are also cleaned up.
When in doubt, please run the `make clean-cloud` with the command below or contact the authors to avoid unnecessary costs.

<!-- Make clean cloud cell -->

### Basic Test (AWS, 30 minutes)
To test that your local machine is configured properly and that you have access to the AWS resources, you can run the following command (with `doe-suite` as working directory):
```shell
make run suite=audit_fairness id=new
```
which will launch two sets of servers on AWS to reproduce the fairness experiments (Fig. 6, column 1).
The command will run the experiments, fetch the results and store them in the `doe-suite-results` folder.
> Tip: To debug problems it can be helpful to comment out the line `stdout_callback = community.general.selective` in [doe-suite/ansible.cfg](../doe-suite/ansible.cfg).

### Basic Test (Local, 15 minutes)
We also provide a `Makefile` in the project's root directory to run scripts locally.
For this, we require MP-SPDZ dependencies to be installed, which can be summarized as the following:
```shell
# Ubuntu (this might try show some prompts, so its good to run this in a real terminal window)
sudo apt-get install -y automake build-essential clang cmake git libboost-dev libboost-iostreams-dev libboost-thread-dev libgmp-dev libntl-dev libsodium-dev libssl-dev libtool python3 libomp-dev libboost-filesystem-dev

# MacOS
brew install openssl boost libsodium gmp yasm ntl cmake libomp
```
This should install all dependencies necessary for local execution, including the dependencies of MP-SPDZ.
We refer to the [MP-SPDZ documentation](https://github.com/data61/MP-SPDZ?tab=readme-ov-file#tldr-source-distribution) for more details on those dependencies.

To install the framework's dependencies in the project directory, run:
```shell
make install
```
Note: If you encounter `ld: library not found for -lomp` from the linker, it may help to add the following to `MP-SPDZ/CONFIG.mine`
```
# These were tested for MacOS
MY_CFLAGS += -I/usr/local/opt/openssl/include -I/opt/homebrew/opt/openssl/include -I/opt/homebrew/include -I/usr/local/opt/libomp/include
MY_LDLIBS += -L/usr/local/opt/openssl/lib -L/opt/homebrew/lib -L/opt/homebrew/opt/openssl/lib -L/usr/local/opt/libomp/lib
```

**Datasets** You must store the relevant datasets in the `MP-SPDZ/Player-Data` directory.
We provide the datasets from the paper, preprocessed to work with MP-SPDZ,
in a public s3 bucket at `http://arc-mpspdz-data.s3.amazonaws.com/{DATASET_NAME}.zip`.
Available datasets are: `adult_3p`, `mnist_full_3party`, `cifar_alexnet_3party`.
For the QNLI dataset, the identifier is `glue_qnli_bert` but the data will be loaded by the compilation script so there is no need to download it.
```shell
mkdir -p MP-SPDZ/Player-Data
wget http://arc-mpspdz-data.s3.amazonaws.com/adult_3p.zip -O adult_3p.zip
unzip -o adult_3p.zip -d MP-SPDZ/Player-Data/adult_3p
```
Then run one of the tasks with the following command:
```shell
poetry run make ring script=inference dataset=adult_3p
```
The framework will compile the script, compile the MP-SPDZ binaries (which will take several minutes) and then run the script.
<details>
<summary>Expected Output (last 10 lines)</summary>

```
Time = 0.013872 seconds 
Time97 = 0.000278 seconds (0 MB, 0 rounds)
Time98 = 1e-06 seconds (0 MB, 0 rounds)
Time99 = 0.000659 seconds (0.023816 MB, 3 rounds)
Time101 = 0.012051 seconds (0.011441 MB, 148 rounds)
Data sent = 0.035257 MB in ~151 rounds (party 0 only; use '-v' for more details)
Global data sent = 0.058875 MB (all parties)
This program might benefit from some protocol options.
Consider adding the following at the beginning of your code:
        program.use_trunc_pr = True
```
</details>

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Evaluation Workflow

We assume that the following steps are followed within the JupyterLab environment.
This artifact relies on the [DoE-Suite](https://github.com/nicolas-kuechler/doe-suite) to manage experiment configurations,
orchestrate the execution of experiments, collect the results and post-process them into tables and plots.

The doe-suite can be run using a set of commands defined in a [Makefile](doe-suite/Makefile) in the `doe-suite` directory that invoke Ansible.
Use `make run` to run experiments, from now on referred to as _suites_, that are defined in the [doe-suite-config/designs](doe-suite-config/designs) directory.
Results of these experiments are then combined together into plots that are defined in the [doe-suite-config/super_etl](doe-suite-config/super_etl) directory.

For each result shown in the paper, we have a separate section that contains:

1. Code to create and display the plot shown in the paper and corresponding dataframe based on the output files from the benchmarks (stored in `doe-suite-results-cameraready`).
   These files can be downloaded from this polybox:
   ```shell
   wget https://polybox.ethz.ch/index.php/s/U6mfbqch0pmSg9U/download -O doe-suite-results-cameraready.zip
   unzip -o doe-suite-results-cameraready.zip -d .
   ```

2. The command to reproduce the results on AWS. You can uncomment the command and run the cell with Ctrl + Enter.
   Due to the large amount of output and long running time, we recommend to run these commands in a separate terminal window.
   The results from these experiments will be stored in `doe-suite-results` and will appear in a separate set of plots in the notebook.

   **Note that the shell commands to execute benchmarks are by default commented out. Uncomment to initiate the runs for these benchmarks.** (remove `#` before or `"""` enclosing the shell command)

   JupyterLab code cells are blocking which means that when executing a cell (e.g., run a benchmark), we cannot run another cell until the previous cell finished. As a result, it might be to better for long running commands to copy the shell command (excluding the comment and !) and execute it in a Jupyter terminal.

   Finally, while running a benchmark in a cell or a terminal, keep the JupyterLab session open and ensure that the internet connection is stable (for AWS).


### Major Claims
- **(C1)**: Arc instantiated with our consistency protocol is up to 10^4x faster and 10^6x more concise than hashing-based (SHA3) and homomorphic commitment-based (PED) approaches.
- **(C2)**: Across all settings, Arc significantly outperforms related approaches in terms of runtime, with a storage overhead comparable to the hash-based approach.

Both claims are proven by the experiments in Section 6: **Training** (E1, Fig. 4), **Inference** (E2, Fig. 5) and **Auditing** (E3, Fig. 6).

### Experiments

For each of training, inference and auditing, we provide a table of suites that belong to this setting.
Each row contains a rough estimate of the maximum duration of running that suite.
This estimate is based on the raw runtimes in the benchmark logs, but the estimation of the overhead of creating and provisioning the machines may not be completely accurate.
To run a suite, simply select a suite from the table and invoke doe-suite to run it.
We provide an option to run the suite inline, or in a separate terminal window.
> Note: The experiments in the WAN setting are particularly time-consuming. As the only difference between the LAN and WAN settings is the network latency (configured using `tc`),
we recommend running the LAN version of the suites if the goal is to verify the framework's functionality.

> Note: There is a small chance an experiment gets stuck. If the experiment is stuck longer than the estimated time (also indicated by the machines have low CPU and network utilization), please terminate the experiment and try again.

> Note: The machine orchestrating the experiments continuously checks if experiments are still running on the remote servers.
> Should this lose internet connection, it is possible to resume an existing experiment using `make run suite=<<SUITE_NAME>> id=<<ID>>` where `<<ID>>` refers the experiment id or `last` for the most recent experiment of the suite.

### Helper files
Please run these cells to ensure that the helper functions to plot the results are loaded.
<!-- Helper methods -->

### Storage
At this point we also copy the storage.csv from the camera-ready benchmarking files to the AWS results folder for convenience.
This file contains the storage costs for the client for the different consistency protocols for the training and inference phases.
The storage.csv can be recomputed using [notebooks/storage.ipynb](notebooks/storage.ipynb).
<!-- Copy storage -->


<!-- Logging --> 


### Training
The training experiments use the MP-SPDZ training script defined in [training.mpc](scripts/training.mpc) which loads the input, trains the model and then outputs it
while ensuring the correct metadata is output for the consistency scripts.
The results are postprocessed using the ETL pipeline defined in [train.yml](doe-suite-config/super_etl/train.yml).

<details>
<summary>Available Suites</summary>

| Suite                                                                 | Description                                  | Est. Duration |
|-----------------------------------------------------------------------|----------------------------------------------|---------------|
| [train_3pc](doe-suite-config/designs/train_3pc.yml)                   | Training for Adult, MNIST and CIFAR-10 (LAN) | 8h            |
| [train_3pc_wan](doe-suite-config/designs/train_3pc_wan.yml)           | Training for Adult, MNIST and CIFAR-10 (WAN) | 14h           |
| [train_3pc_bert](doe-suite-config/designs/train_3pc_bert.yml)         | Training for QNLI (LAN)                      | 8h            |
| [train_3pc_bert_wan](doe-suite-config/designs/train_3pc_bert_wan.yml) | Training for QNLI (WAN)                      | 8h            |

</details>

### Inference
The inference experiments use the MP-SPDZ inference script defined in [inference.mpc](scripts/inference.mpc).
The results are postprocessed using the ETL pipeline defined in [inference.yml](doe-suite-config/super_etl/inference.yml).

<details>
    <summary>Available Suites</summary>

| Suite                                                                         | Description                                                        | Est. Duration |
|-------------------------------------------------------------------------------|--------------------------------------------------------------------|------------|
| [inference_3pc](doe-suite-config/designs/inference_3pc.yml)                   | Inference for Adult, MNIST and CIFAR-10 (LAN)                      | 2h         |
| [inference_3pc_wan](doe-suite-config/designs/inference_3pc_wan.yml)           | Inference for Adult, MNIST and CIFAR-10 (WAN)                      | 6h30m      |
| [inference_3pc_bert](doe-suite-config/designs/inference_3pc_bert.yml)         | Inference for QNLI (LAN)                                           | 3h         |
| [inference_3pc_bert_wan](doe-suite-config/designs/inference_3pc_bert_wan.yml) | Inference for QNLI (WAN)                                           | 6h         |

</details>



### Auditing
The inference experiments use the MP-SPDZ scripts for auditing defined in [scripts](scripts), starting with `audit_`.
The results are postprocessed using the ETL pipeline defined in [audit.yml](doe-suite-config/super_etl/audit.yml).

The following experiments are available to run:
<details>
    <summary>Available Suites</summary>

| Suite                                                                               | Description                                                        | Est. Duration |
|-------------------------------------------------------------------------------------|--------------------------------------------------------------------|---------------|
| [audit_robustness](doe-suite-config/designs/audit_robustness.yml)                   | Robustness function (Sec. 5.1) for Adult, MNIST and CIFAR-10 (LAN) | 2h            |
| [audit_robustness_wan](doe-suite-config/designs/audit_robustness_wan.yml)           | Robustness function (Sec. 5.1) for Adult, MNIST and CIFAR-10 (WAN) | 6h            |
| [audit_knnshapley](doe-suite-config/designs/audit_knnshapley.yml)                   | KNNShapley function (Sec. 5.2) for Adult, MNIST and CIFAR-10 (LAN) | 6h            |
| [audit_knnshapley_wan](doe-suite-config/designs/audit_knnshapley_wan.yml)           | KNNShapley function (Sec. 5.2) for Adult, MNIST and CIFAR-10 (WAN) | 17h           |
| [audit_fairness](doe-suite-config/designs/audit_fairness.yml)                       | Fairness function (Sec 5.1) for Adult, MNIST, and CIFAR-10 (LAN)   | 30m           |
| [audit_fairness_wan](doe-suite-config/designs/audit_fairness_wan.yml)               | Fairness function (Sec 5.1) for Adult, MNIST, and CIFAR-10 (WAN)   | 2h40m         |
| [audit_shap](doe-suite-config/designs/audit_shap.yml)                               | SHAP function (Sec. 5.3) for Adult, MNIST, and CIFAR-10 (LAN)      | 2h            |
| [audit_shap_wan](doe-suite-config/designs/audit_shap_wan.yml)                       | SHAP function (Sec. 5.3) for Adult, MNIST, and CIFAR-10 (WAN)      | 8h            |
| [audit_knnshapley_bert](doe-suite-config/designs/audit_knnshapley_bert.yml)         | KNNShapley function (Sec. 5.2) for QNLI (Semi-honest, LAN)         | 4h            |
| [audit_knnshapley_bert_mal](doe-suite-config/designs/audit_knnshapley_bert_mal.yml) | KNNShapley function (Sec. 5.2) for QNLI (Malicious, LAN)           | 6h            |
| [audit_knnshapley_bert_wan](doe-suite-config/designs/audit_knnshapley_bert_wan.yml) | KNNShapley function (Sec. 5.2) for QNLI (Semi-honest, WAN)         | 2h            |

</details>



### Storage Overhead
We compute the storage costs for Fig. 4 & Fig. 5 using [storage.ipynb](notebooks/storage.ipynb)
which stores the result in a csv in `doe-suite-results` which is loaded by the pipeline.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### MNIST (Appendix, Extended Version)
The MNIST experiments are part of the training, inference and auditing suites, but are displayed in a separate plot in the appendix of the extended version due to space reasons. Therefore, we have a separate ETL pipeline for these results.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->

[poetry-shield]: https://img.shields.io/badge/poetry-grey?style=for-the-badge&logo=poetry
[poetry-url]: https://python-poetry.org/


[doesuite-shield]: https://img.shields.io/badge/doe--suite-grey?style=for-the-badge&logo=github
[doesuite-url]: https://github.com/nicolas-kuechler/doe-suite


[aws-shield]: https://img.shields.io/badge/aws-ec2-grey?style=for-the-badge&logo=amazonaws
[aws-url]: https://aws.amazon.com/


[euler-shield]: https://img.shields.io/badge/ethz-euler-grey?style=for-the-badge
[euler-url]: https://scicomp.ethz.ch/wiki/Euler