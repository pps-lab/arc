# Cryptographic Auditing MPC

> An evaluation framework to evaluate arbitrary auditing functions.

This repository contains the evaluation framework and the auditing function prototype
developed by Alexander Mandt as part of his Bachelor Thesis *Cryptographic Auditin in ML*.

## Background

The evaluation framework builds upon [MP-SPDZ](https://github.com/data61/MP-SPDZ/tree/master) and the 
[Design of Experiments Suite](https://github.com/nicolas-kuechler/doe-suite/tree/main) and a basic
unterstanding of both frameworks is recommended before using this evaluation framework.

Concretely, the evaluation framework builds upon MP-SPDZ as its Mulit-Party Computation runtime. 
Auditing functions that will be evaluated by the evaluation framework must use the MP-SPDZ-specific 
Python-DSL for developing auditing functions. Please see  [the MP-SPDZ documentation](https://mp-spdz.readthedocs.io/en/latest/) 
for more information about the DSL.

Please note that MP-SPDZ and the DoE suite are integrated as submodules of this git repository and the 
commits referenced `mp-spdz` and `doe-suite` define the version of MP-SPDZ and the DoE suite currently used by the evaluation framework
respectively.

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

In this section, we will go through all steps to setup the evaluation framework for use.

1. Ensure that Python 3.9 and Git is installed on the local machine
2. Checkout this Github repository into a Folder of your liking: (For this section, we checkout into the `mpc-audit` folder)
   ```
   git checkout --recursive git@github.com:pps-lab/cryptographic-auditing-mpc.git mpc-audit
   ```
   The `--recursive` option ensures that all submodules in the folder are checked out together with the evaluation framework.

3.  Go into the folder in which the repository has been checked out into
    ```
    cd mpc-audit
    ```
4. Setup the **Design of Experiments Suite**:
   1. Install Poetry [(See Instructions)](https://python-poetry.org/docs/).
   2. Create a `key pair for AWS` in the region `eu-central-1` (Frankfurt) [(see instructions)](https://docs.aws.amazon.com/servicecatalog/latest/adminguide/getstarted-keypair.html). Ensure that you store your AWS Key Pair in a folder from which you can access it.
   Storing the AWS Key Pair in `~/.ssh/` is recommended. (For this section, we assume that we store our key pair `aws-keys.pem` in 
   folder `~/.ssh/`, so the full path is `~/.ssh/aws-keys.pem`)
   3. Ensure that you can checkout Github repositories with SSH (should already be done)
   4. Go to `doe-suite` subfolder housing the *Designs of Experiments Suite* root:
      ```
      cd doe-suite
      ```
   5. Now within the  *Designs of Experiments Suite* root, we install all requirements:
      ```
      poetry install
      ```
   6. Configure `ssh` and `ssh-agent`:
      
      1. Configure `~/.ssh/config`. We create `~/.ssh/config`, if it does not exist, and add an entry to this file so that we can easily connect to EC2 instances via SSH:
         ```
         Host ec2*
             IdentityFile <Path to AWS Key Pair>
             User ubuntu
             ForwardAgent yes
         ``` 
         For our example, the entry would look as follows:
         ```
         Host ec2*
             IdentityFile ~/.ssh/aws-keys.pem
             User ubuntu
             ForwardAgent yes
         ```
      2. We now add the Github private key to `ssh-agent`. This allows cloning a GitHub repository on an EC2 instance
         without copying the private key or entering credentials. The process depends on your environment but should be
         as follows [(see Instructions)](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) (Note that only the `Adding your SSH key to the ssh-agent` section is interesting)

         Please note that in a later part of this Getting Started guide, we prepare a `setup-env.local.sh` script that
         takes care of setting up an instance of `ssh-agent` and loading all keys into that instance . So we do not really do anything here. 
      
   7. Install the AWS CLI version 2 and configure Boto:
      + Install AWS CLI version 2 [(see Instructions)](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
      + Configure AWS credentials for Boto [(see Instructions)](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html)
        ```
        aws configure
        ```

        For this command, you must enter the AWS Access Key Id and your AWS Access Key Secret for the AMI user you will be using
        to interact with AWS. Also, please configure as the standard region `eu-central-1` (the Frankfurt Region) and as default
        output `json` (as the evaluation framework was tested with this setting)
   
   8. Install the required Ansible collections:
      ```
      poetry run ansible-galaxy install -r requirements-collection.yml
      ``` 
   9. Step out of the `doe-suite` folder:
      ```
      cd ..
      ``` 

5. (Optional) Install MP-SPDZ locally to run scripts locally:
   1. Step into the `mp-spdz` folder
      ```
      cd mp-spdz
      ```


   2. We first ensure that all local dependecies are installed:
      
      For Ubuntu 20.04, executing the following command is enough to install all dependencies:
      ```
      sudo apt-get install automake build-essential git libboost-dev libboost-thread-dev libntl-dev libsodium-dev libssl-dev libtool m4 python3 texinfo yasm
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
   2. Generate the initial base Image with the AMI generator [(See instructions under the Usage Section)](./ansible-utils/ami-generator/README.md) and note the image id of the generated base image.

   3. Prepare the `setup-env.local.sh` environment setup script:

      This script must be loaded into the terminal before using the evaluation framework. 
      It sets all the needed environment variables, prepares the `ssh-agent` and provides convenience commands to run the
      evaluation framework

      1. Use the template file `template-setup-env.local.sh.default` file to create a `setup-env.local.sh` file:
         ```
         cp template-setup-env.localh.sh.default setup-env.local.sh
         ``` 
      2. Then, open the `setup-env.local.sh` file in your favourite editor and enter the path to your
         Github SSH keys and your AWS SSH Keys into the environment variables `GITHUB_KEYS` and `AWS_KEYS` respectively.
      
      3. To setup the environment, we need to load the `setup-env.local.sh` from the evaluation framework root as follows:
         ```
         source setup-env.local.sh
         ```

   4. Configure the evaluation framework
         

      


    

             


