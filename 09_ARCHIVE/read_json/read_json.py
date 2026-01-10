import json


with open("config.json", "r") as f:
    config = json.load(f)

print(f"config: {type(config)}")