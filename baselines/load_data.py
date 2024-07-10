import json

def get_json_files(path):
    file = open(path)
    json_file = json.load(file)
    return json_file