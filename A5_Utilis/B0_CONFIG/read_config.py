import json

###################################################
###               Read Config                   ###
###################################################
def read_config(path_config):

    with open(path_config, "r") as f:
        config = json.load(f)

    return config