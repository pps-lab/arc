import subprocess

subprocess.run(
    " ".join(["./compile.py", "-C", "-D", "-R", "64", "../scripts/test-assign.mpc"]),
    shell=True,
    cwd="./mp-spdz"
)