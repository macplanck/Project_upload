from pathlib import Path
from A5_Utilis.B0_CONFIG.read_config import read_config
from A5_Utilis.B2_PERF.perf import PERF

###################################################
###                   CLASS                     ###
###################################################
class global_param:
    def __init__(self, path_conf):
        self.config = read_config(path_conf)
        self.token_bit = self.config["token_size"];  
        self.weight_bit = self.config["weight_size"];  
        self.token_size = 2 ** self.token_bit
        self.seq_length  = self.config["seq_length"]
        self.hidden_size = self.config["hidden_size"]
        self.LUT_NUM = self.config["LUT_NUM"]
        self.head_dim = self.config["head_dim"]
        self.head_num = self.config["head_num"]
        self.intermediate_size = self.config["intermediate_size"]

###################################################
###            GLOBAL PARAMETERS                ###
###################################################
BASE = Path(__file__).resolve().parent
if PERF:
    PATH_CONF = (BASE / "../B2_PERF/config_param.json")
else:
    PATH_CONF = (BASE / "config_param.json")
param = global_param(PATH_CONF)


