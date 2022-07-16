import pydantic
import enum
import typing

class ProtocolChoices(enum.Enum):
    EMULATE_X = "emulate_env"
    REPLICATED_RING_PARTY_X = "semi_honest_3"

class TaskConfig(pydantic.BaseModel):
    player_id: int
    sleep_time: float
    player_count: int
    player_0_hostname: str
    abs_path_to_code_dir: str
    # MPC specific options
    script_name: str
    script_args: list[str]
    protocol_setup: ProtocolChoices
    input_file_name: str
    result_dir: str

class ArgumentLineConfig(pydantic.BaseModel):
    player_id: int
    sleep_time: float

class JsoncMpcConfig(pydantic.BaseModel,extra=pydantic.Extra.forbid):
    player_count: int
    player_0_hostname: str
    abs_path_to_code_dir: str
    script_name: str
    script_args: list[str]
    protocol_setup: ProtocolChoices
    input_file_name: str


class JsonConfigModel(pydantic.BaseModel,extra=pydantic.Extra.ignore):
    mpc: JsoncMpcConfig



def parse_json_config(config_path):
    config_obj = JsonConfigModel.parse_file(config_path)
    return config_obj

def build_task_config(json_cofig_obj: JsonConfigModel, player_number: int, 
    sleep_time: float, result_dir: str):
    conf_obj = TaskConfig(
        player_id=player_number,
        sleep_time=sleep_time,
        player_count=json_cofig_obj.mpc.player_count,
        player_0_hostname=json_cofig_obj.mpc.player_0_hostname,
        abs_path_to_code_dir=json_cofig_obj.mpc.abs_path_to_code_dir,
        protocol_setup=json_cofig_obj.mpc.protocol_setup,
        script_args=json_cofig_obj.mpc.script_args,
        script_name=json_cofig_obj.mpc.script_name,
        input_file_name=json_cofig_obj.mpc.input_file_name,
        result_dir=result_dir
    )
    return conf_obj
