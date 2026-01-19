from A5_Utilis.B0_CONFIG.read_config import read_config

###################################################
###                   CLASS                     ###
###################################################
class global_param:
    def __init__(self, path_conf):
        self.config = read_config(path_conf)
        token_size = self.config["token_size"];  
        self.token_size = 2 ** token_size
        self.seq_length  = self.config["seq_length"]
        self.hidden_size = self.config["hidden_size"]
        self.LUT_NUM = self.config["LUT_NUM"]

###################################################
###            GLOBAL PARAMETERS                ###
###################################################
PATH_CONF = "config_param.json"
param = global_param(PATH_CONF)


