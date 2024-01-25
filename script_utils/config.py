from typing import Optional

from Compiler.script_utils import output_utils

from pydantic import BaseModel


def from_program_args(program_args, ModelClass):
    params = output_utils.parse_kv_args(program_args)

    # TODO: could verify that ModelClass inherits from BaseAuditModel

    model = ModelClass(**params)

    print(f"Compiling with Config\n  {model}")

    if model.debug:
        print("WARNING: THE CURRENT PROGRAM REVEALS DEBUG OUTPUT")

    return model


class BaseAuditModel(BaseModel):

    debug: bool = False
    emulate: bool = False
    dataset: str = None
    batch_size: int = 128
    n_threads: int = 36
    trunc_pr: bool = False
    round_nearest: bool = False

    audit_trigger_idx: int = 0

    # Whether to check parties' inputs for consistency
    consistency_check: Optional[str] = "pc" # also "sha3" and "cerebro" for cerebro
    sha3_approx_factor: int = 1
    cerebro_output_approx_factor: int = 1

    # only run n_batches, works for some auditing functions only
    n_batches: int = 0

    class Config:
        extra = "allow"
        smart_union = True