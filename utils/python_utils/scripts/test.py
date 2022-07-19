import subprocess
import os
my_env = os.environ.copy()
my_env["MY_PATH"] = "SIMPLE_PATH"


subprocess.run(
    " ".join(["./compile.py -C -D -R 64 ../scripts/test-assign.mpc"]),
    shell=True,
    cwd="./mp-spdz",
    env=my_env
)

subprocess.run(
    " ".join(["./Scripts/ring.sh test-assign"]),
    shell=True,
    cwd="./mp-spdz",
    env=my_env
)