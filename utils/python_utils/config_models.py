import pydantic

class TaskConfig(pydantic.BaseModel):
    player_id: int
    sleep_time: float
    player_count: int
    player_0_hostname: str
    abs_path_to_code_dir: str
    # MPC specific options
    script_name: str
    script_args: list[str]


