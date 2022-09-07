# AMI Generator

This folder contains the AMI Generator. The AMI Generator generates the base machine image for the experiment hosts 
for the evaluation framework. 

## Usage

The AMI Generator requires Ansible to be usable. The AMI Generator relies on the Ansible installation that 
comes as part of the *Design of Experiments Suite* virtual python environment. To use the AMI Generator, the
following steps need to be completed:

0. Load the *Design of Experiments Suite* virtual python environment (we assume that we start from the evaluation framework root directory):
   ```
   cd doe-suite
   poetry shell
   cd ..
   ``` 

   Then, go to the AMI Geneator folder (from the evaluation framework root folder):
   ```
   cd ansible-utils/ami-generator
   ```
1. Configure the AMI Generator variables in file `site2_vars.yml` as required. 
2. Create the server that will be used to generate the base image (the *base image server*):
   ```
   ansible-playbook create_aws.yml
   ```

3. Initiate the AMI Generation:
   ```
   ansible-playbook create_mp_spdz_ami.yml
   ```

   This step will now connect to the base image server and configure the base image server in such a way that the 
   final configuration of the base image server can be used as the base state of each experiment host.
   After the configuration of the base image server has been completed, this playbook will signal AWS to stop the base image server
   and create an Amazon Machine Image based on the final state of the base image server.
   The Playbook will conclude its execution before the AMI generation may have finished. Therefore, you must observe the
   AMI generation in the AWS management console. You can find the status of the AMI [here](https://eu-central-1.console.aws.amazon.com/ec2/home?region=eu-central-1#Images:visibility=owned-by-me). You may have to login into the AWS Console. 
   If you only have programmatic access, you can use the AWS v2 CLI to request the current status of the AMI as follows:
   ```
   aws ec2 describe-images --filters "Name=name,Values=<Name of the AMI as defined in site2_vars.yml>" | grep "status"
   ```
   If the status is available, then the AMI image has been successfully generated.

4. Once the AMI is available, we decommision the base image server
   ```
   ansible-playbook destroy_aws.yml
   ```

5. We now obtain the image id via the following AWS CLI v2 command:
   ```
   aws ec2 describe-images --filters "Name=name,Values=<Name of the AMI as defined in site2_vars.yml>" | grep "ImageId"
   ```
   The obtained image id must be noted down, since it will be needed during the re-configuration of the evaluation framework. 

6. After we are done with the AMI generator, we can leave the *Design of Experiments Suite* virtual python environment by:
   ```
   exit
   ```

## `site2_var.yml` Variables

In this section, we describe each variable in `site2_vars.yml` and its effects. We denote with the *base image server* the 
server that the AMI Generator creates to generate the base machine image.

The `site2_vars.yml` file contains the following variables:

+ **test_host_name**: This is the name of the base image server.
+ **aws_target_profile**: This is the name of the profile that will be used to access Amazon Web Services. This option is only needed if one works with multiple AWS CLI profiles. It is not needed to change this value during normal usage.
+ **aws_security_group**: This is the name of the security group that the base image server will be assigned to.
+ **aws_key_name**: The name of the key pair that will be used to configure the SSH login for the base image server. It is recommended to use the same key pair as used by the evaluation framework.
+  **aws_instance_type**: This is the name of the instance type of the base image server. Testing has shown that only the `t2.2xlarge` and the `t2.xlarge` instance types can be used for the base image server under the `T2` general purpose EC2 instance types. For the compute optimized `C5` instances, the `c5.9xlarge` instance type was also tested. Other instance types should work too, if they provide similar or superior specs as the `t2.2xlarge` instance types. 
+ **target_repo**: This variable contains the SSH address of the evaluation framework. Please adapt this value, if another Git repository houses the evaluation framework.
+ **git_repo_dest**: This is the path to the folder into which the `target_repo` git repository will be cloned into. This path should not be changed unless necessary.
+ **git_repo_mpspdz_location**: This variable contains the path to the MP-SPDZ root folder of the evaluation framework. This variable
should not be changed, unless the folder structure of the evaluation framework has been changed.
+ **make_job_nums**: The number of parallel jobs that should be executed by `make`. This setting is used by the AMI Generator to
set how many parallel jobs can be executed by the `make` stages of the base image generation. The `make` stages involve the 
compilation of the prerequisite MP-SPDZ libraries and the compilation of all MP-SPDZ virtual machines.
+ **aws_ami_name**: This is the name of the final base image AMI. Please change this value if the old base images should be retained.
This is because each base image must have a unique AMI name. The AMI Generator will abort the base image generation if an AMI 
with the name specified in `aws_ami_name` already exists.