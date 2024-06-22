

from ruamel.yaml import YAML
import os


def get_share_length(path):

    content = get_share_data(path)
    if content is None:
        return 0

    print("Format", content)

    p_total_length = 0
    for p in content:
        p_total_length += p["length"]

    return p_total_length

def get_share_data(path):
    if not os.path.exists(path):
        print("File does not exist", path)
        return None
    yaml = YAML()
    with open(path, 'r') as file:
        content = yaml.load(file)
    return content

def get_total_share_length(format_dir, player_count):
    player_input_list = []
    for i in range(player_count):
        filename = 'Input-Binary-P%d-0-format' % i
        path = os.path.join(format_dir, filename)
        data = get_share_data(path)
        player_input_list.append(data)

    filename = "Output-format"
    path = os.path.join(format_dir, filename)
    output_data = get_share_data(path)
    total_output_length = 0
    if output_data is not None:
        for p in output_data:
            total_output_length += p["length"]

    return player_input_list, output_data, total_output_length