"""This module defines the code that executes the MP-SPDZ facilities such as the MPC protocol virtual machines and the
MPC program compiler. It defines abstractions to ease the integration of MPC protocol virtual machines.

The main abstraction is the BaseRunner class. It defines a standarized way to execute a program using the subprocessing module
without having to directly interact with the subprocessing module. It povides methods that must be implemented so that a program
can be run.

The CompilerArguments enum class holds the default compiler arguments that need to be applied to MPC scripts at compile-time
for each MPC protocol vm backend supported by the evaluation framework. These flags are important because the MP-SPDZ compiler
must know in advance, in which computational domain the MPC script will be compiled under. And since different MPC
VMs might have different computational domains. So, the CompilerArguments enum class holds for each Protocol Choice the
appropriate Compiler arguments. Please note that  these argumenets must be kept in sync with the values provided in the
ProtocolChoices enum class in the config_def.py module.

The CompilerRunner class acts as the high-level interface to the MP-SPDZ compiler. It executes the 'compile.py' compiler and
compiles the provided script with given arguments and compiler flags.

The ScriptBaseRunner class is a more specific abstract class that builds on the BaseRunner abstract class and
provides the base abstraction for any class that implements the interface to an MP-SPDZ protocol virtual machine.
The ScriptBaseRunner provides base parameters and a base implementation of the environment hook for Script Runners to
avoid duplicate code.


The EmulatorRunner, ReplicatedRingPartyRunner, BrainPartyRunner, ReplicatedBinPartyRunner, PsReplicatedBinPartyRunner,
ShamirPartyRunner, MaliciousShamirPartyRunner are the concrete high-level interfaces to the MP-SPDZ-specific protocol virtual machines
'./emulate.x', './replicated-ring-party.x', './brain-party.x', './replicated-bin-party.x', './ps-rep-bin-party.x', './shamir-party.x',
'./malicious-shamir-party.x'.

The ProtocolRunners enum maps each ProtocolChoices enum entry to the corresponding Runner class.

To implement additional protocols, the following steps need to be followed:

1. Extend the ProtocolChoices and CompilerArguments enums with a uniquely named enum entry. Please ensure that both enums are kept in sync during this extension. The ProtocolChoices enum will define how the protocol choice will be named for the protocol choice parameter in the experiment template. The CompilerArguments enum entry will decide which compiler arguments will be given to the protocol choice. Note that argument spacing is done by separating list entries

2. Implement the ScriptBaseRunner, ensuring that the class follows the following naming convention to keep the look of the classes similar: <Name of ProtocolChoicesEntry>Runner. It is important that the _program() and _args() arguments are implemented. More information can be found in the documentation of each respective class.

3. Extend the ProtocolRunners enum to include the new implementation of the ScriptBaseRunner. Please ensure that the ProtoclRunners enum is synced with the ProtocolChoices and CompilerArguments enums


The module provides the following functionalities:

- function script_name_and_args_to_correct_execution_name(script_name, script_args): str
    Tranforms the script_name and the script_args argument list into the name of the bytecode file that an MPC protocol virtual machine executes.

- class BaseRunner:
    The main abstraction to implement the execution of MP-SPDZ facilities via a standardizes interface to the subprocessing module.

- enum CompilerArguments:
    Provides a named mapping between the concrete MPC protocol VM runners and required compiler flags for MPC script compilation for the given MPC protocol.

- class CompilerRunner:
    Provides the high-level interface to interact with the MP-SPDZ compiler.

- class ScriptBaseRunner:
    Provides the more specific interface abstraction to the MP-SPDZ protocol VMs.

- class EmulatorRunner:
    Is the high-level interface to './emulate.x'

- class ReplicatedRingPartyRunner:
    Is the high-level interface to './replicated-ring-party.x'

- class BrainPartyRunner:
    Is the high-level interface to './brain-party.x'

- class ReplicatedBinPartyRunner:
    Is the high-level interface to './replicated-bin-party.x'

- class PsReplicatedBinPartyRunner
    Is the high-level interface to './ps-rep-bin-party.x'

- class ShamirPartyRunner:
    Is the high-level interface to './shamir-party.x'

- class MaliciousShamirPartyRunner:
    Is the high-level interface to './malicious-shamir-party.x'

- enum ProtocolRunners:
    Provides a named mapping between the concrete MPC protocol VM runners and the interface implementations for each of the concrete MPC protocol VMs.

"""
import abc
import subprocess
import enum
import os
import shlex
from typing import Dict, List


# This class expects that each Runner is executed
def program_args_cmdline(program_args: Dict[str, str]) -> List[str]:
    if program_args is None:
        return []
    list_tuples = [(f'-{k}', v) for k, v in program_args.items()]
    # flatten
    flat_list = [item for sublist in list_tuples for item in sublist]
    return flat_list

def script_name_and_args_to_correct_execution_name(script_name, script_args):
    """Tranforms the script_name and the script_args argument list into the name of the bytecode file that an MPC protocol virtual machine executes.

    Parameters
    ----------
    - script_name : str
        Name of the given script
    - script_args : list[str]
        List of arguments with which the given script was compiled under.

    Returns
    -------
    The name of the executable file generated by the comiler and is expected by the MPC protocol VM to identify the executable
    """
    serialized_args = [f'{k}__{v}' for k, v in script_args.items()]
    return f"{'-'.join([script_name] + serialized_args)}"


class BaseRunner(abc.ABC):
    """The main abstraction to implement the execution of MP-SPDZ facilities via a standardizes interface to the subprocessing module.

    To extend the base runner, the _program(), _args() and the _env() arguments need to be implemented. These methods provide the path to the program that should be run, the arguments that the specific program should run under, and the environment variables that the program should be started with. More can be found in the documentation of each method.

    Methods
    -------
    - _program():
        Returns the path to the program that should be executed.
    - _args():
        Returns a list of arguments that should be provided to the program under the path returned by _program()
    - _env():
        Returns the set of environment variables under which the program with path provided by _program() should be run under

    - run():
        Execute the program with path provided by _program() with arguments provided by _args() and with environment provided by _env()
    """

    @property
    def program(self):
        return str(self._program())

    @abc.abstractmethod
    def _program(self):
        pass

    @property
    def args(self):
        args_filtered = filter(lambda x: len(x) > 0, self._args())
        return [str(s) for s in args_filtered]

    @abc.abstractmethod
    def _args(self):
        pass

    @property
    def env(self):
        return self._env()

    @abc.abstractmethod
    def _env(self):
        pass

    def run(self, stdout=None, stderr=None):
        subprocess.run(
            " ".join([self.program] + self.args),
            shell=True,
            cwd="./MP-SPDZ/",
            check=True,
            capture_output=False,
            env=self.env,
            stdout=stdout,
            stderr=stderr
        )


class CompilerArguments(enum.Enum):
    """Provides a named mapping between the concrete MPC protocol VM runners and required compiler flags for MPC script compilation for the given MPC protocol."""

    # These are only used if no compiler_args are specific.
    EMULATE_X = ['-R', '64']
    REPLICATED_RING_PARTY_X = ['-R', "64"]
    REP4_RING_PARTY_X = ['-R', "64", '-Z', '4', '-C']  # edabits  -> need -C for vectorization
    BRAIN_PARTY_X = ['-R', '64']
    REPLICATED_BIN_PARTY_X = ['-B', '64']
    PS_REP_BIN_PARTY_X = ['-B', '64']
    SHAMIR_PARTY_X = ["-F", "64"]
    MALICIOUS_SHAMIR_PARTY_X = ["-F", "64"]
    ATLAS_PARTY_X = ["-F", "64"]
    MAL_ATLAS_PARTY_X = ["-F", "64"]
    REP_FIELD_PARTY = ["-F", "64"]
    MAL_REP_FIELD_PARTY = ["-F", "64"]
    MAL_REP_RING_PARTY = ["-R", "64"]

    SY_REP_RING_PARTY = ['-R', "64"]
    SY_REP_FIELD_PARTY = ['-F', "64"]
    PS_REP_FIELD_PARTY = ['-F', "64"]
    SPDZ2K_PARTY = ['-R', "64"]
    SEMI2K_PARTY = ['-R', "64"]
    SEMI_PARTY = ['-F', "128"]
    MASCOT_PARTY = ['-F', "64"]
    MASCOT_OFFLINE = ['-F', "64"]

class CompilerRunner(BaseRunner):
    """Provides the high-level interface to interact with the MP-SPDZ compiler."""

    def __init__(self, script_name, script_args, compiler_args, code_dir):
        """
        Parameters
        ----------
        - script_name : str
            Name of the script that should be compiled
        - script_args : list[str]
            List of space-separated arguments under which the given script should be compiled under
        - compiler_args : list[str]
            List of space-separated  compiler arguments with which the given script should be compiled with
        - code_dir : str
            The absolute path to the directory containing the root of the evaluation framework code
        """
        self._script_name = script_name
        self._script_args = script_args
        self._compiler_args = compiler_args
        self._code_dir = code_dir

    def _program(self):
        return "./compile.py"

    def _env(self):
        my_env = os.environ.copy()
        if "PYTHONPATH" in my_env.keys():
            my_env["PYTHONPATH"] = f"{my_env['PYTHONPATH']}:{os.path.join(self._code_dir, 'scripts/')}"
        else:
            my_env["PYTHONPATH"] = f"{os.path.join(self._code_dir,'scripts/')}"
        return my_env


    def _args(self):
        serialized_args = [f'{k}__{v}' for k, v in self._script_args.items()]
        return self._compiler_args + \
            [os.path.join(self._code_dir, "scripts", f"{self._script_name}.mpc")] \
             + serialized_args


class ScriptBaseRunner(BaseRunner):
    """Provides the more specific interface abstraction to the MP-SPDZ protocol VMs.

    To extend this class, only the _program() and _args() methods need to be implemented, as a default _env() implementation is
    already provided with this class. This is because the MP-SPDZ protocol VMs do not require the setting of environment variables.
    Please also note that this class provides the set of all attributes that may be needed by any of the given MPC protocol VMs. However,
    not every MPC protocol VM will need every attribute. Please consult the documentation of each MPC protocol VM to see which attributes
    are needed.

    Attributes
    ----------
    - output_prefix : str
        The output prefix of the raw text output files of the MPC protocol VM
    - script_name : str
        The name of the script that should be executed
    - script_args : list[str]
        The list of arguments under which the given script was compiled under
    - playere_0_host : str
        The hostname of the machine that hosts the player 0 MPC protocol VM proccess
    - player_id : int
        The id of the MPC protocol VM process
    - player_count : int
        The number of MPC protocol VM processes that will be part of the experiment execution
    """
    def __init__(self, output_prefix, script_name, args, player_0_host, player_id, custom_prime, custom_prime_length, player_count, program_args):
        """
        Parameters:
        - output_prefix : str
            The output prefix of the raw text output files of the MPC protocol VM
        - script_name : str
            The name of the script that should be executed
        - args : list[str]
            The list of arguments under which the given script was compiled under
        - player_0_host : str
            The hostname of the machine that hosts the player 0 MPC protocol VM proccess
        - player_id : int
            The id of the MPC protocol VM process
        - player_count : int
            The number of MPC protocol VM processes that will be part of the experiment execution
        - program_args: list
            List of additional program arguments that should be passed to the MPC protocol VM
        """
        self.output_prefix = output_prefix
        self.script_name = script_name
        self.script_args = args
        self.player_0_host = player_0_host
        self.player_id = player_id
        self.player_count = player_count
        self.custom_prime = custom_prime
        self.custom_prime_length = custom_prime_length
        assert not (self.custom_prime is not None and self.custom_prime_length is not None),\
            "It is not possible to specify a custom prime AND a custom prime length!"
        self.program_args = program_args

    def _env(self):
        my_env = os.environ.copy()
        return my_env


class EmulatorRunner(ScriptBaseRunner):
    """Is the high-level interface to './emulate.x'"""
    def _program(self):
        return "./emulate.x"

    def _args(self):
        return ["-OF", self.output_prefix,
            script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)]



class ReplicatedRingPartyRunner(ScriptBaseRunner):
    """Is the high-level interface to './replicated-ring-party.x'"""
    def _program(self):
        print("Run ReplicatedRingPartyRunner")
        return "./replicated-ring-party.x"

    def _args(self):
        program_args_flat = program_args_cmdline(self.program_args)
        return ["-OF", self.output_prefix,
            "-h", f"{self.player_0_host}",
            "-pn", "12300",
                "-v",
            f"{self.player_id}"] + program_args_flat + [
                script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
            ]


class Replicated4RingPartyRunner(ScriptBaseRunner):
    """Is the high-level interface to './rep4-ring-party.x'"""
    def _program(self):
        print("Run Replicated4RingPartyRunner")
        return "./rep4-ring-party.x"

    def _args(self):
        return ["-OF", self.output_prefix,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-v",
                f"{self.player_id}",
                script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)]


class BrainPartyRunner(ScriptBaseRunner):
    """Is the high-level interface to './brain-party.x'"""
    def _program(self):
        print("Run  BrainPartyRunner")
        return "./brain-party.x"

    def _args(self):
        return ["-OF", self.output_prefix,
            "-h", f"{self.player_0_host}",
            "-pn", "12300",
                "-v",
            f"{self.player_id}",
            script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)]


class ReplicatedBinPartyRunner(ScriptBaseRunner):
    """Is the high-level interface to './replicated-bin-party.x'"""
    def _program(self):
        print("Run ReplicatedBinPartyRunner")
        return "./replicated-bin-party.x"

    def _args(self):
        return ["-OF", self.output_prefix,
            "-h", f"{self.player_0_host}",
            "-pn", "12300",
                "-v",
            f"{self.player_id}",
            script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)]

class PsReplicatedBinPartyRunner(ScriptBaseRunner):
    """Is the high-level interface to './ps-rep-bin-party.x'"""
    def _program(self):
        print("Run PsReplicatedBinPartyRunner")
        return "./ps-rep-bin-party.x"

    def _args(self):
        return ["-OF", self.output_prefix,
            "-h", f"{self.player_0_host}",
            "-pn", "12300",
                "-v",
            f"{self.player_id}",
            script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
        ]

class SyReplicatedRingPartyRunner(ScriptBaseRunner):
    """Is the high-level interface to './sy-rep-bin-party.x'"""
    def _program(self):
        print("Run SyReplicatedRingPartyRunner")
        return "./sy-rep-ring-party.x"

    def _args(self):
        return ["-OF", self.output_prefix,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-v",
                f"{self.player_id}",
                script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
                ]

class SyReplicatedFieldPartyRunner(ScriptBaseRunner):
    """Is the high-level interface to './sy-rep-bin-party.x'"""
    def _program(self):
        print("Run SyReplicatedFieldPartyRunner")
        return "./sy-rep-field-party.x"

    def _args(self):
        program_args_flat = program_args_cmdline(self.program_args)
        custom_prime_arg = f"-P {self.custom_prime}" if self.custom_prime is not None else ""
        custom_prime_length_arg = f"-lgp {self.custom_prime_length}" if self.custom_prime is not None else ""
        return ["-OF", self.output_prefix,
                custom_prime_arg, custom_prime_length_arg,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-v",
                f"{self.player_id}"] + program_args_flat + [
                script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
                ]


class PsReplicatedFieldPartyRunner(ScriptBaseRunner):
    """Is the high-level interface to './sy-rep-bin-party.x'"""
    def _program(self):
        print("Run PsReplicatedFieldPartyRunner")
        return "./ps-rep-field-party.x"

    def _args(self):
        custom_prime_arg = f"-P {self.custom_prime}" if self.custom_prime is not None else ""
        custom_prime_length_arg = f"-lgp {self.custom_prime_length}" if self.custom_prime is not None else ""
        return ["-OF", self.output_prefix,
                custom_prime_arg, custom_prime_length_arg,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-v",
                f"{self.player_id}",
                script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
                ]

class ShamirPartyRunner(ScriptBaseRunner):
    """Is the high-level interface to './shamir-party.x'"""
    def _program(self):
        return "./shamir-party.x"

    def _args(self):
        custom_prime_arg = f"-P {self.custom_prime}" if self.custom_prime is not None else ""
        custom_prime_length_arg = f"-lgp {self.custom_prime_length}" if self.custom_prime is not None else ""
        return ["-OF", self.output_prefix,
            custom_prime_arg, custom_prime_length_arg,
            "-h", f"{self.player_0_host}",
            "-pn", "12300",
                "-v",
            "-N", f"{self.player_count}",
            f"{self.player_id}",
            script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)]

class MaliciousShamirPartyRunner(ScriptBaseRunner):
    """Is the high-level interface to './malicious-shamir-party.x'"""
    def _program(self):
        return "./malicious-shamir-party.x"

    def _args(self):
        custom_prime_arg = f"-P {self.custom_prime}" if self.custom_prime is not None else ""
        custom_prime_length_arg = f"-lgp {self.custom_prime_length}" if self.custom_prime is not None else ""
        return ["-OF", self.output_prefix,
                custom_prime_arg, custom_prime_length_arg,
            "-h", f"{self.player_0_host}",
            "-pn", "12300",
                "-v",
            "-N", f"{self.player_count}",
            f"{self.player_id}",
            script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
        ]

class AtlasPartyRunner(ScriptBaseRunner):
    """Is the high-level interface to './shamir-party.x'"""
    def _program(self):
        return "./atlas-party.x"

    def _args(self):
        custom_prime_arg = f"-P {self.custom_prime}" if self.custom_prime is not None else ""
        custom_prime_length_arg = f"-lgp {self.custom_prime_length}" if self.custom_prime is not None else ""
        return ["-OF", self.output_prefix,
                custom_prime_arg, custom_prime_length_arg,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-v",
                "-N", f"{self.player_count}",
                f"{self.player_id}",
                script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)]

class MaliciousAtlasPartyRunner(ScriptBaseRunner):
    """Is the high-level interface to './shamir-party.x'"""
    def _program(self):
        return "./mal-atlas-party.x"

    def _args(self):
        custom_prime_arg = f"-P {self.custom_prime}" if self.custom_prime is not None else ""
        custom_prime_length_arg = f"-lgp {self.custom_prime_length}" if self.custom_prime is not None else ""
        return ["-OF", self.output_prefix,
                custom_prime_arg, custom_prime_length_arg,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-v",
                "-N", f"{self.player_count}",
                f"{self.player_id}",
                script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)]

class ReplicatedFieldPartyRunner(ScriptBaseRunner):
    """Is the high-level interface to './shamir-party.x'"""
    def _program(self):
        return "./replicated-field-party.x"

    def _args(self):
        program_args_flat = program_args_cmdline(self.program_args)
        custom_prime_arg = f"-P {self.custom_prime}" if self.custom_prime is not None else ""
        custom_prime_length_arg = f"-lgp {self.custom_prime_length}" if self.custom_prime is not None else ""
        return ["-OF", self.output_prefix,
                custom_prime_arg, custom_prime_length_arg,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-v",
                f"{self.player_id}"] + program_args_flat + [
                script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
        ]

class MaliciousReplicatedFieldPartyRunner(ScriptBaseRunner):
    """Is the high-level interface to './shamir-party.x'"""
    def _program(self):
        return "./malicious-rep-field-party.x"

    def _args(self):
        program_args_flat = program_args_cmdline(self.program_args)
        custom_prime_arg = f"-P {self.custom_prime}" if self.custom_prime is not None else ""
        custom_prime_length_arg = f"-lgp {self.custom_prime_length}" if self.custom_prime is not None else ""
        return ["-OF", self.output_prefix,
                custom_prime_arg, custom_prime_length_arg,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-v",
                f"{self.player_id}"] + program_args_flat + [
            script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
        ]

class MaliciousReplicatedRingPartyRunner(ScriptBaseRunner):
    """Is the high-level interface to './shamir-party.x'"""
    def _program(self):
        return "./malicious-rep-ring-party.x"

    def _args(self):
        program_args_flat = program_args_cmdline(self.program_args)
        return ["-OF", self.output_prefix,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-v",
                f"{self.player_id}"] + program_args_flat + [
            script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
        ]


class MascotPartyRunner(ScriptBaseRunner):
    """Is the high-level interface to './malicious-shamir-party.x'"""
    def _program(self):
        return "./mascot-party.x"

    def _args(self):
        custom_prime_arg = f"-P {self.custom_prime}" if self.custom_prime is not None else ""
        custom_prime_length_arg = f"-lgp {self.custom_prime_length}" if self.custom_prime is not None else ""
        return ["-OF", self.output_prefix,
                custom_prime_arg, custom_prime_length_arg,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-N", f"{self.player_count}",
                "-v",
                f"{self.player_id}",
                script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
                ]

class SemiPartyRunner(ScriptBaseRunner):
    def _program(self):
        return "./semi-party.x"

    def _args(self):
        custom_prime_arg = f"-P {self.custom_prime}" if self.custom_prime is not None else ""
        custom_prime_length_arg = f"-lgp {self.custom_prime_length}" if self.custom_prime is not None else ""
        return ["-OF", self.output_prefix,
                custom_prime_arg, custom_prime_length_arg,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-N", f"{self.player_count}",
                "-v",
                f"{self.player_id}",
                script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
                ]

class LowgearPartyRunner(ScriptBaseRunner):
    """Is the high-level interface to './malicious-shamir-party.x'"""
    def _program(self):
        return "./lowgear-party.x"

    def _args(self):
        program_args_flat = program_args_cmdline(self.program_args)
        custom_prime_arg = f"-P {self.custom_prime}" if self.custom_prime is not None else ""
        custom_prime_length_arg = f"-lgp {self.custom_prime_length}" if self.custom_prime is not None else ""
        return (["-OF", self.output_prefix,
                custom_prime_arg, custom_prime_length_arg,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-N", f"{self.player_count}",
                "-v",
                f"{self.player_id}"]
                + program_args_flat + [
                    script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
                ])

class HighgearPartyRunner(ScriptBaseRunner):
    """Is the high-level interface to './malicious-shamir-party.x'"""
    def _program(self):
        return "./highgear-party.x"

    def _args(self):
        custom_prime_arg = f"-P {self.custom_prime}" if self.custom_prime is not None else ""
        custom_prime_length_arg = f"-lgp {self.custom_prime_length}" if self.custom_prime is not None else ""
        return ["-OF", self.output_prefix,
                custom_prime_arg, custom_prime_length_arg,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-N", f"{self.player_count}",
                "-v",
                f"{self.player_id}",
                script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
                ]

class MascotOfflineRunner(ScriptBaseRunner):
    def _program(self):
        return "./mascot-offline.x"

    def _args(self):
        return ["-OF", self.output_prefix,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-N", f"{self.player_count}",
                f"{self.player_id}",
                script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
                ]

class ProtocolRunners(enum.Enum):
    """Provides a named mapping between the concrete MPC protocol VM runners and the interface implementations for each of the concrete MPC protocol VMs."""
    EMULATE_X = EmulatorRunner
    REPLICATED_RING_PARTY_X = ReplicatedRingPartyRunner
    REP4_RING_PARTY_X = Replicated4RingPartyRunner
    BRAIN_PARTY_X=BrainPartyRunner
    REPLICATED_BIN_PARTY_X=ReplicatedBinPartyRunner
    PS_REP_BIN_PARTY_X=PsReplicatedBinPartyRunner
    SHAMIR_PARTY_X=ShamirPartyRunner
    ATLAS_PARTY_X=AtlasPartyRunner
    MAL_ATLAS_PARTY_X=MaliciousAtlasPartyRunner
    MALICIOUS_SHAMIR_PARTY_X=MaliciousShamirPartyRunner
    REP_FIELD_PARTY=ReplicatedFieldPartyRunner
    MAL_REP_FIELD_PARTY=MaliciousReplicatedFieldPartyRunner
    MAL_REP_RING_PARTY=MaliciousReplicatedRingPartyRunner
    SY_REP_RING_PARTY=SyReplicatedRingPartyRunner
    SY_REP_FIELD_PARTY=SyReplicatedFieldPartyRunner
    PS_REP_FIELD_PARTY=PsReplicatedFieldPartyRunner
    MASCOT_PARTY=MascotPartyRunner
    SEMI_PARTY=SemiPartyRunner
    MASCOT_OFFLINE=MascotOfflineRunner
    LOWGEAR_PARTY=LowgearPartyRunner
    HIGHGEAR_PARTY=HighgearPartyRunner
