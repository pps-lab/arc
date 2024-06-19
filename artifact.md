
# Artifact Appendix
<a name="readme-top"></a>

This repository contains the artifact for the paper ["Holding Secrets Accountable: Auditing Privacy-Preserving Machine Learning"](https://arxiv.org/abs/2402.15780).

Some more text about the main components:



### Arc Components
- Cryptographic Auditing MPC
- Consistency Check Protocol
!TODO here: How to reference these external repo's? What about MP-SPDZ and DOE-suite?




<!-- ABOUT THE PROJECT -->
## Description & Requirements

We provide the necessary commands to reproduce the entire evaluation of the paper.
The evaluation is built using the [DoE-Suite](https://github.com/nicolas-kuechler/doe-suite), making it the most straightforward way to reproduce the results.
However, it is also possible to obtain the individual commands used to invoke the framework and run them manually.

### Built With

* [![Poetry][poetry-shield]][poetry-url]
* [![DoE-Suite][doesuite-shield]][doesuite-url]
* [![AWS][aws-shield]][aws-url]

### Security, privacy, and ethical concerns
> [!WARNING]
> Executing experiments on your AWS infrastructure involves the creation of EC2 instances, resulting in associated costs.
> It is important to manually check that any created machines are terminated afterward.

### How to access

### Hardware dependencies

### Software dependencies

### Benchmarks


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Set-up
We provide a [JupyterLab environment]() to run the experiments and evaluate the artifact of Zeph. Our JupyterLab environment contains further documentation, runnable code cells to run benchmarks locally or on AWS (using `!` to run shell commands), and runnable code cells to recreate the plots from the paper.

To execute a selected cell, either click on the run button in the toolbar or use `Ctrl + Enter`. To keep the JupyterLab readable, cells containing longer code sections to create plots are initially collapsed (indicated by three dots). However, by clicking on the dots, the cell containing the code expands (`View/Collapse Selected Code` to collapse the code again).

**Note that the shell commands to execute benchmarks are by default commented out. Uncomment to initiate the runs for these benchmarks.** (remove `#` before or `"""` enclosing the shell command)

#### AWS
AWS login information can be found in the Artifact submission system. This will allow the Artifact reviewers to run the evaluation on the same resources stated in the paper submission. The experiments on AWS are automated with ansible-playbooks and can be called from the JupyterLab environment.

**Please ensure that the AWS resources are cleaned up if they are not used in the artifact**
If the playbook terminates normally (i.e., without error and is not interrupted), then the playbook ensures that the created resources are also cleaned up. 
When in doubt, please run the `make clean-cloud` with the command below or contact the authors to avoid unnecessary costs.


<!-- GETTING STARTED -->
### Installation

To get a local copy up and running follow these simple steps.

* [Python Poetry](https://python-poetry.org/)
    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```

* [Make](https://www.gnu.org/software/make/)

* Local clone of the repository (with submodules)
    ```sh
    git clone --recurse-submodules git@github.com:pps-lab/arc.git
    ```

#### DoE-Suite

1. Setup environment variables for the Doe-Suite:
    ```sh
    # root project directory (expects the doe-suite-config dir in this folder)
    export DOES_PROJECT_DIR=<PATH>

    #  Your unique short name, such as your organization's acronym or your initials.
    export DOES_PROJECT_ID_SUFFIX=<SUFFIX>
    ```

   For AWS EC2:
    ```sh
    export DOES_CLOUD=aws

    # name of ssh key used for setting up access to aws machines (name of key not path)
    export DOES_SSH_KEY_NAME=<YOUR-PRIVATE-SSH-KEY-FOR-AWS>
    ```

2. Set up SSH Config and for AWS setup AWS CLI.
   Currently, the `doe-suite` is configured to use the AWS region `eu-central-1`.
   For more details refer to the [doe-suite documentation](https://nicolas-kuechler.github.io/doe-suite/installation.html#base-installation).

#### Benchmarking files
1Download and store the benchmarking output files from the camera ready submission:
```shell
wget https://pps-mpspdz-data.s3.amazonaws.com/doe-suite-results-cameraready.zip
unzip doe-suite-results-cameraready.zip -d doe-suite-results-cameraready
```

[!Tip]
To debug problems it can be helpful to comment out the line `stdout_callback = community.general.selective` in [doe-suite/ansible.cfg](../doe-suite/ansible.cfg).

#### Testing the installation
From this point on it is possible to run the Jupyter notebook which contains the experiments and evaluation of the artifact.
To launch the Jupyter notebook in the environment with the correct dependencies, we provide a make command.
1. Navigate to the `doe-suite` sub-directory and run `make jupyter` which will launch these instructions in the form of a notebook.
```shell
cd doe-suite
make jupyter
```
Now run the first cell,
```python

```
and check that it prints `Environment loaded successfully`.

### Basic Test (AWS, 30 minutes)
To test that your local machine is configured properly and that you have access to the AWS resources, you can run the following command:
```shell
make run suite=audit_fairness id=new
```
which will launch two sets of servers on AWS to reproduce the fairness experiments (Fig. 6, column 1).

### Basic Test (Local)
We also provide a `Makefile` to run scripts locally.
First install the framework's dependencies in the project directory:
```shell
make install
```
Note: On MacOS, if you encounter `ld: library not found for -lomp` from the linker, it may help to add the following to `MP-SPDZ/CONFIG.mine`
```
MY_CFLAGS += -I/usr/local/opt/openssl/include -I/opt/homebrew/opt/openssl/include -I/opt/homebrew/include -I/usr/local/opt/libomp/include
MY_LDLIBS += -L/usr/local/opt/openssl/lib -L/opt/homebrew/lib -L/opt/homebrew/opt/openssl/lib -L/usr/local/opt/libomp/lib
```
You must store the relevant datasets that have been preprocessed to work with MP-SPDZ in the `MP-SPDZ/Player-Data` directory.
The datasets are available in a public s3 bucket at `http://pps-mpspdz-data.s3.amazonaws.com/{DATASET_NAME}.zip`.
Available datasets are: `adult_3p`, `mnist_full_3party`, `cifar_alexnet_3party`.
For the QNLI dataset, the identifier is `glue_qnli_bert` but the data will be loaded by the compilation script so there is no need to download it.
```shell
# Download the dataset
wget http://pps-mpspdz-data.s3.amazonaws.com/adult_3p.zip -O MP-SPDZ/Player-Data/adult_3p.zip
unzip MP-SPDZ/Player-Data/adult_3p.zip -d MP-SPDZ/Player-Data/adult_3p
```
And then run one of the tasks with the following command:
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

## Evaluation Workflow

We assume that the following steps are followed within the JupyterLab environment. 
This artifact relies on the [DoE-Suite](https://github.com/nicolas-kuechler/doe-suite) to manage experiment configurations,
orchestrate the execution of experiments, collect the results and post-process them into tables and plots.

The doe-suite can be run using a set of commands defined in a [Makefile](doe-suite/Makefile) that invoke Ansible.
Use `make run` to run experiments (suites) that are defined in the `doe-suite-config/designs` directory.
Results of these experiments are then combined together into plots that are defined in the `doe-suite-config/super_etl` directory.

For each result shown in the paper, we have a separate section that contains:

1. Code to create and display the plot shown in the paper and corresponding dataframe based on the output files from the benchmarks (stored in `doe-suite-results-cameraready`)

2. The command to reproduce the results on AWS. You can uncomment the command and run the cell with Ctrl + Enter. 
Due to the large amount of output and long running time, we recommend to run these commands in a separate terminal window.

Note that for improved readability, the code for creating the table and the plot is initially collapsed but can be openend by clicking on the three dots. To collapse the code again, select the cell by clicking on it and then go to View/Collapse Selected Code.

JupyterLab code cells are blocking which means that when executing a cell (e.g., run a benchmark), we cannot run another cell until the previous cell finished. As a result, it might be to better for long running commands to copy the shell command (excluding the comment and !) and execute it in a Jupyter terminal.

In any case, while running a benchmark in a cell or a terminal, keep the JupyterLab session open and ensure that the internet connection is stable (for AWS).

### Major Claims
- **(C1)**: Arc instantiated with our consistency protocol is up to 10^4x faster and 10^6x more concise than hashing-based (SHA3) and homomorphic commitment-based (PED) approaches.
- **(C2)**: Across all settings, Arc significantly outperforms related approaches in terms of runtime, with a storage overhead comparable to the hash-based approach

Both claims are proven by the experiments in Section 6 **Training** (E1, Fig. 4), **Inference** (E2, Fig. 5) and **Auditing** (E3, Fig. 6).

### Experiments
For each of training, inference and auditing, we provide a list of suites that belong to this setting.
Each suite contains a rough estimate of the duration of running the suite.
This estimate is based on the raw runtimes in the benchmark logs, but the estimation of the overhead of creating and provisioning the machines may not be completely accurate.
> Note: The experiments in the WAN setting are particularly time-consuming. As the only difference between the LAN and WAN settings is the network latency (configured using `tc`), 
we recommend running the LAN version of the suites if the goal is to verify the framework's functionality.


### Storage Overhead
We compute the storage costs in [storage.ipynb](notebooks/storage.ipynb). 

### Training
The training experiments use the MP-SPDZ training script defined in [training.mpc](scripts/training.mpc) which loads the input, trains the model and then outputs it
while ensuring the correct metadata is output for the consistency scripts.

<details>
<summary>Available Suites</summary>

| Suite              | Description                                  | Est. Duration |
|--------------------|----------------------------------------------|---------------|
| train_3pc          | Training for Adult, MNIST and CIFAR-10 (LAN) | 2h            |
| train_3pc_wan      | Training for Adult, MNIST and CIFAR-10 (WAN) | 12h           |
| train_3pc_bert     | Training for QNLI (LAN)                      | ??            |
| train_3pc_bert_wan | Training for QNLI (WAN)                      | ??            |

</details>

### Inference

The following experiments are available:
<details>
    <summary>Available Suites</summary>

| Suite                     | Description                                                        | Est. Duration |
|---------------------------|--------------------------------------------------------------------|---------------|
| inference_3pc             | Inference for Adult, MNIST and CIFAR-10 (LAN)                      | 1h            |
| inference_3pc_wan         | Inference for Adult, MNIST and CIFAR-10 (WAN)                      | 5h30m         |
| inference_3pc_bert        | Inference for QNLI (LAN)                                           | 1h40m         |
| inference_3pc_bert_wan    | Inference for QNLI (WAN)                                           | 6h            |

</details>



### Auditing

15:04 start

The following experiments are available to run:
<details>
    <summary>Available Suites</summary>

| Suite                     | Description                                                        | Est. Duration |
|---------------------------|--------------------------------------------------------------------|---------------|
| audit_robustness          | Robustness function (Sec. 5.1) for Adult, MNIST and CIFAR-10 (LAN) | 1h            |
| audit_robustness_wan      | Robustness function (Sec. 5.1) for Adult, MNIST and CIFAR-10 (WAN) | 6h            |
| audit_knnshapley          | KNNShapley function (Sec. 5.2) for Adult, MNIST and CIFAR-10 (LAN) | 6h            |
| audit_knnshapley_wan      | KNNShapley function (Sec. 5.2) for Adult, MNIST and CIFAR-10 (WAN) | 17h           |
| audit_fairness            | Fairness function (Sec 5.1) for Adult, MNIST, and CIFAR-10 (LAN)   | 30m           |
| audit_fairness_wan        | Fairness function (Sec 5.1) for Adult, MNIST, and CIFAR-10 (WAN)   | 2h40m         |
| audit_shap                | SHAP function (Sec. 5.3) for Adult, MNIST, and CIFAR-10 (LAN)      | 1h            |
| audit_shap_wan            | SHAP function (Sec. 5.3) for Adult, MNIST, and CIFAR-10 (WAN)      | 8h            |
| audit_knnshapley_bert     | KNNShapley function (Sec. 5.2) for QNLI (Semi-honest, LAN)         | 1h ??         |
| audit_knnshapley_bert_mal | KNNShapley function (Sec. 5.2) for QNLI (Malicious, LAN)           | 4h ??         |
| audit_knnshapley_bert_wan | KNNShapley function (Sec. 5.2) for QNLI (Semi-honest, WAN)         | 1.5h ??       |

</details>


## Notes on Reusability

# Notes / TODO
- import ruamel.yaml error, does it appear on non my machines?
- 



<!-- MARKDOWN LINKS & IMAGES -->

[poetry-shield]: https://img.shields.io/badge/poetry-grey?style=for-the-badge&logo=poetry
[poetry-url]: https://python-poetry.org/


[doesuite-shield]: https://img.shields.io/badge/doe--suite-grey?style=for-the-badge&logo=github
[doesuite-url]: https://github.com/nicolas-kuechler/doe-suite


[aws-shield]: https://img.shields.io/badge/aws-ec2-grey?style=for-the-badge&logo=amazonaws
[aws-url]: https://aws.amazon.com/


[euler-shield]: https://img.shields.io/badge/ethz-euler-grey?style=for-the-badge
[euler-url]: https://scicomp.ethz.ch/wiki/Euler