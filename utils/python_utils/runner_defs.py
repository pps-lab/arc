import abc
import subprocess
import enum
import os


# This class expects that each Runner is executed 

class BaseRunner(abc.ABC):

    @property
    def program(self):
        return str(self._program())

    @abc.abstractmethod
    def _program(self):
        pass

    @property
    def args(self):
        return [str(s) for s in self._args()]

    @abc.abstractmethod
    def _args(self):
        pass

    @property
    def env(self):
        return self._env()

    @abc.abstractmethod
    def _env(self):
        pass

    def run(self):
        subprocess.run(
            " ".join([self.program] + self.args),
            shell=True,
            cwd="./mp-spdz/",
            check=True,
            capture_output=False,
            env=self.env
        )


class CompilerArguments(enum.Enum):
    EMULATE_X = ['-R', '64']
    REPLICATED_RING_PARTY_X = ['-R', "64"]

class CompilerRunner(BaseRunner):

    def __init__(self, script_name, script_args, compiler_args, code_dir):
        self._script_name = script_name
        self._script_args = script_args
        self._compiler_args = compiler_args
        self._code_dir = code_dir
    
    def _program(self):
        return "./compile.py"

    def _env(self):
        my_env = os.environ.copy()
        my_env['PYTHONPATH'] = os.path.join(self._code_dir,"scripts/")
        return my_env


    
    def _args(self):
        return self._compiler_args + \
            [f"../scripts/{self._script_name}.mpc"] + \
            self._script_args


class ScriptBaseRunner(BaseRunner):
    def __init__(self, output_prefix, script_name, args, player_0_host, player_id):
        self.output_prefix = output_prefix
        self.script_name = script_name
        self.script_args = args
        self.player_0_host = player_0_host
        self.player_id = player_id
    
    def _env(self):
        my_env = os.environ.copy()
        return my_env


class EmulatorRunner(ScriptBaseRunner):
    
    def _program(self):
        return "./emulate.x"
    
    def _args(self):
        return ["-OF", self.output_prefix, 
            f"{self.script_name}-{'-'.join([str(s) for s in self.script_args])}"]



class ReplicatedRingPartyRunner(ScriptBaseRunner):
    
    def _program(self):
        print("Run ReplicatedRingPartyRunner")
        return "./replicated-ring-party.x"

    def _args(self):
        return ["-OF", self.output_prefix,
            "-h", f"{self.player_0_host}",
            "-pn", "12300",
            f"{self.player_id}",
            f"{self.script_name}-{'-'.join([str(s) for s in self.script_args])}"]



class ProtocolRunners(enum.Enum): 
    EMULATE_X = EmulatorRunner
    REPLICATED_RING_PARTY_X = ReplicatedRingPartyRunner




    
