import abc
import subprocess
import enum


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

    def run(self):
        subprocess.run(
            " ".join([self.program] + self.args),
            shell=True,
            cwd="./mp-spdz/",
            check=True
        )


class CompilerArguments(enum.Enum):
    EMULATE_X = ['-R', '64']

class CompilerRunner(BaseRunner):

    def __init__(self, script_name, script_args, compiler_args):
        self._script_name = script_name
        self._script_args = script_args
        self._compiler_args = compiler_args
    
    def _program(self):
        return "./compile.py"
    
    def _args(self):
        return self._compiler_args + \
            [f"../scripts/{self._script_name}.mpc"] + \
            self._script_args

class EmulatorRunner(BaseRunner):

    def __init__(self, output_prefix, script_name, args):
        self._out_prefix = output_prefix
        self._script_name = script_name
        self._script_args = args
    
    def _program(self):
        return "./emulate.x"
    
    def _args(self):
        return ["-OF", self._out_prefix, 
            f"{self._script_name}-{'-'.join([str(s) for s in self._script_args])}"]




    
