# Cryptographic Auditing MPC

> An evaluation framework to evaluate arbitrary auditing functions.

This repository contains the evaluation framework and the auditing function prototype
developed by Alexander Mandt as part of his Bachelor Thesis *Cryptographic Auditing in ML*.

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

1. Ensure that Python 3.9 and Git is installed on the local machine
2. Checkout this Github repository into a Folder of your liking: (For this section, we checkout into the `mpc-audit` folder)
   ```
   git clone --recursive git@github.com:pps-lab/cryptographic-auditing-mpc.git mpc-audit
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
      poetry install # or poetry update if the previous command fails
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


   2. Prepare the `setup-env.local.sh` environment setup script:

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
   3. Generate the initial base Image with the AMI generator [(See instructions under the Usage Section)](./ansible-utils/ami-generator/README.md) and note the image id of the generated base image.

   

   4. Reseed the server types with the new image:
      1. We start the configuration by running the repotemplate script as follows (from the evaluation framework root):
         ```
         cd doe-suite
         poetry run python src/scripts/repotemplate.py
         ```

      2. Next, we need to define the list of server types to generate.

         + The evaluation framework has 4 default server types that must be generated:
           + *hserv1*: A `c5.9xlarge` instance and the type of the server that executes the main MPC virtual machine
           + *hserv2*: A `c5.9xlarge` instance and the type of the server(s) that execute the additional MPC virtual machines
           + *server*: A `t2.2xlarge` instance and the type of the server that executes the main MPC virtual machine
           + *server2*: A `t2.2xlarge` instance and the type of the server(s) that execute the additonal MPC virtual machines
         
         + During the repo reconfiguration, the following must be done:
           + For the initial configuration, we need to regenerate the `all` group vars. But for later configuration,
             you can skip the regeneration of the `all` group vars.

             The following values must be set during the `all` group vars configuration:
             ```
             Unique Project ID: <A unique project id> # This is the name of the project
             Git Remote Repository: <The SSH address of the Git repository containing the evaluation framework code> # This variable defines the location where the Git repository is located, that contains the experiment code (here the evaluation framework)
             Playbook number of tries to check for experiment finished: <A number up to 1000, 300 is recommended>
             Time to wait between checking in seconds: <A number, 10 is recommended>
             AWS Key name: <Name of the Private key pair that you created for AWS>
             ```
           + Do not generate the default host types, but generate the above 4 host types in the given order. You can also
             generate additional host types. We will describe how to add additonal host types in a later section.
           + For the given host types, the following settings must be honored:
             + **hserv1, hserv2**: 
               ```
               EC2 Instance type: c5.9xlarge
               EC2 Volume size in GB: <your choice, but at least 40 GB>
               EC2 image AMI: <The image id you noted down before>
               Snapshot ID for this instance: <Leave empty, choose the default given by the script>
               ```
             + **server,server2**:
               ```
               EC2 Instance type: t2.2xlarge
               EC2 Volume size in GB: <your choice, but at least 40 GB>
               EC2 image AMI: <The image id you noted down before>
               Snapshot ID for this instance: <Leave empty, choose the default given by the script>
               ```
   5. Prepare the Input AWS Bucket:
      + We need to create an AWS S3 bucket in which all input data will be stored for the evaluation framework.
        [(See Instructions here)](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html).
        
      + Once the AWS Input Bucket was created, we need to give the name of the bucket to the evaluation framework.
        We do this by adapting the preparation step of the evaluation framework that downloads the data from the input bucket.
        To achieve this, we need to modify the `mlexample_bucket_name` variable in the `does_config/roles/ml-example-setup-1/vars/main.yml`
        ```
        ---
        mlexample_bucket_name: <The name of your AWS input bucket>
        mlexample_object_name: ...
        mlexample_tmp_storage: ...
        mlexample_input_dest: ...
        ...
        ```
      
      + Please note that AWS instances need to have permissions to access the input buckets. For the evaluation framework, it is
        recommended to create an instance role (like 'EC2WithS3') and grant that role the necessary permissions to access the AWS 
        input bucket. Then, please add the following line to the particular section outlined below in the file 
        `doe-suite/src/roles/suite-aws-ec2-create/tasks/main.yml`:
        
        ```
        ...
        #######################################################################
        # Create EC2 Instances for all host_types
        ######################################################################
        
        - name: Create EC2 Instances (only assign subset of tags yet)
          community.aws.ec2_instance:
            instance_type: '{{ ec2config.instance_type }}'
            key_name: '{{ ec2config.key_name }}'
            image_id: '{{ ec2config.ec2_image_id }}'
            region: '{{ ec2config.aws_region }}'
            security_group: '{{ ec2config.sg_name }}'
            vpc_subnet_id: '{{ ec2config.vpc_subnet_id }}'
            instance_role: EC2WithS3Role # <-- This line must be added to the presented section
            wait: no
            purge_tags: yes
            network:
         ...
        ```
        This change will make the *Designs of Experiments Suite* create instances with the defined instance role, thereby granting
        each experiment host the permissions needed to work.
        
      
7. After this step, the evaluation framework should be usable. You can test the usability by running the `toy-example` experiment.
   Please note that for the `toy-example` experiment, we need to run the  `utils/python_utils/create_toy_input.py` script to generate
   the input file container for the input files needed for the `toy-example` experiment. Also, this input file container must
   be placed into the AWS Input file bucket before the experiment can be run.

## Local development
We aim to keep the git submodule of MP-SPDZ up-to-date with the main branch.
This means that to execute programs locally in the emulator using MP-SPDZ, 
you need to link the custom program files that exist in the top level repo to MP-SPDZ.

These commands assume you are in the root of the repo:
1. Link scripts: `ln -s  $(PWD)/scripts/audit_mnist.mpc $(PWD)/mp-spdz/Programs/Source/audit_mnist.mpc`
2. Link script_utils `ln -s $(PWD)/script_utils $(PWD)/mp-spdz/Compiler/script_utils`

Then `cd mp-spdz` and use MP-SPDZ as usual:
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









            
           
         

      


    

             


