
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
However, it is also possible to obtain the individual commands used to invoke the [dp-planner](../dp-planner) and run them manually.

> [!WARNING]
> Executing experiments on your AWS infrastructure involves the creation of EC2 instances, resulting in associated costs.
> It is important to manually check that any created machines are terminated afterward.

### Built With

* [![Poetry][poetry-shield]][poetry-url]
* [![DoE-Suite][doesuite-shield]][doesuite-url]
* [![AWS][aws-shield]][aws-url]
* [![ETHZ Euler][euler-shield]][euler-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>




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
    git clone --recurse-submodules git@github.com:pps-lab/cohere.git
    ```

* [TeX Live](https://www.tug.org/texlive/): Only required for reproducing the plots (see e.g., `make plot-all`).
    ```sh
    sudo apt install texlive-full
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

   For ETHZ Euler (Slurm-based Scientific Compute Cluster):
    ```sh
    export DOES_CLOUD=euler

    # Replace <YOUR-NETHZ> with your NETHZ username
    export DOES_EULER_USER=<YOUR-NETHZ>
    ```


2. Set up SSH Config and for AWS setup AWS CLI.
   Currently, the `doe-suite` is configured to use the AWS region `eu-central-1`.
   For more details refer to the [doe-suite documentation](https://nicolas-kuechler.github.io/doe-suite/installation.html#base-installation).


[!Tip]
To debug problems it can be helpful to comment out the line `stdout_callback = community.general.selective` in [doe-suite/ansible.cfg](../doe-suite/ansible.cfg).






<!-- MARKDOWN LINKS & IMAGES -->

[poetry-shield]: https://img.shields.io/badge/poetry-grey?style=for-the-badge&logo=poetry
[poetry-url]: https://python-poetry.org/


[doesuite-shield]: https://img.shields.io/badge/doe--suite-grey?style=for-the-badge&logo=github
[doesuite-url]: https://github.com/nicolas-kuechler/doe-suite


[aws-shield]: https://img.shields.io/badge/aws-ec2-grey?style=for-the-badge&logo=amazonaws
[aws-url]: https://aws.amazon.com/


[euler-shield]: https://img.shields.io/badge/ethz-euler-grey?style=for-the-badge
[euler-url]: https://scicomp.ethz.ch/wiki/Euler