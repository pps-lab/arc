
# Arc
<a name="readme-top"></a>

This repository contains the artifact for the paper ["Holding Secrets Accountable: Auditing Privacy-Preserving Machine Learning"](https://arxiv.org/abs/2402.15780).

Some more text about the main components:

### Arc Components
- Cryptographic Auditing MPC
- Consistency Check Protocol
!TODO here: How to reference these external repo's? What about MP-SPDZ and DOE-suite?




<!-- ABOUT THE PROJECT -->
## About The Project

We provide the necessary commands to reproduce the entire evaluation of the paper.
The evaluation is built using the [DoE-Suite](https://github.com/nicolas-kuechler/doe-suite), making it the most straightforward way to reproduce the results.
However, it is also possible to obtain the individual commands used to invoke the framework and run them manually.

> [!WARNING]
> Executing experiments on your AWS infrastructure involves the creation of EC2 instances, resulting in associated costs.
> It is important to manually check that any created machines are terminated afterward.

### Built With

* [![Poetry][poetry-shield]][poetry-url]
* [![DoE-Suite][doesuite-shield]][doesuite-url]
* [![AWS][aws-shield]][aws-url]
* [![ETHZ Euler][euler-shield]][euler-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Experiments
We provide a [JupyterLab environment]() to run the experiments and evaluate the artifact of Zeph. Our JupyterLab environment contains further documentation, runnable code cells to run benchmarks locally or on AWS (using `!` to run shell commands), and runnable code cells to recreate the plots from the paper.

To execute a selected cell, either click on the run button in the toolbar or use `Ctrl + Enter`. To keep the JupyterLab readable, cells containing longer code sections to create plots are initially collapsed (indicated by three dots). However, by clicking on the dots, the cell containing the code expands (`View/Collapse Selected Code` to collapse the code again).

**Note that the shell commands to execute benchmarks are by default commented out. Uncomment to initiate the runs for these benchmarks.** (remove `#` before or `"""` enclosing the shell command)

### AWS
AWS login information can be found in the Artifact submission system. This will allow the Artifact reviewers to run the evaluation on the same resources stated in the paper submission. The experiments on AWS are automated with ansible-playbooks and can be called from the JupyterLab environment.

**Please ensure that the AWS resources are cleaned up if they are not used in the artifact**
If the playbook terminates normally (i.e., without error and is not interrupted), then the playbook ensures that the created resources are also cleaned up. 
When in doubt, please run the `make clean-cloud` with the command below or contact the authors to avoid unnecessary costs.


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

* [Python Poetry](https://python-poetry.org/)
    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```

* [Make](https://www.gnu.org/software/make/)

* Local clone of the repository (with submodules)
    ```sh
    git clone --recurse-submodules git@github.com:pps-lab/arc.git
    ```

### Installation

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
and check that it prints `Installation successful.`.


## Running Experiments

We assume that the following steps are followed within the JupyterLab environment. 
This artifact relies on the [DoE-Suite](https://github.com/nicolas-kuechler/doe-suite) to manage experiment configurations,
orchestrate the execution of experiments, collect the results and post-process them into tables and plots.

The doe-suite can be run using a set of commands defined in a [Makefile](doe-suite/Makefile) that invoke Ansible.
Use `make run` to run experiments (suites) that are defined in the `doe-suite-config/designs` directory.
Results of these experiments are then combined together into plots that are defined in the `doe-suite-config/super_etl` directory.

For each result shown in the paper, we have a separate section that contains:

1. Code to produce the results in form of a table based on the output files from the benchmarks

2. Code to create the plot shown in the paper from the results table

3. Code that displays both the results and the figure and allows you to switch between the results from the paper and the reproduced results

4. The command to reproduce the results (locally / on aws for microbenchmarks, and on aws for the end-to-end benchmark. You can uncomment the command and run the cell with Ctrl + Enter.

Note that for improved readability, the code for creating the table and the plot is initially collapsed but can be openend by clicking on the three dots. To collapse the code again, select the cell by clicking on it and then go to View/Collapse Selected Code.

JupyterLab code cells are blocking which means that when executing a cell (e.g., run a benchmark), we cannot run another cell until the previous cell finished. As a result, it might be to better for long running commands to copy the shell command (excluding the comment and !) and execute it in a Jupyter terminal.

In any case, while running a benchmark in a cell or a terminal, keep the JupyterLab session open and ensure that the internet connection is stable (for AWS).

### Inference
inference_3pc_bert: 21:26 start, 23:02 end, duration 1h36m


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