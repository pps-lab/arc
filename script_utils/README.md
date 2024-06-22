# Cryptographic Auditing: Script Utils

This folder contains the MPC Utils and allows the introduction of custom Python modules that can be loaded
by MPC scripts in the `scripts` folder under the `Compiler.script_utils` module.

The following modules are provided:

+ `output_utils.py` - This module provides the output facilities that integrate with the Output Processor
+ `ml_modified.py` - This module is an extended version of the original `ml.py` module from MP-SPDZ.
+ `audit_function_utils.py` - This module provides utility functions for auditing functions. It implements functions to find the median 
  of an Array of values, and to compute the absolute difference array of an array for a given value.

More documentation can be found in each Python module.