from pathlib import Path
from A5_Utilis.B0_CONFIG.read_config import read_config

###################################################
###            GLOBAL PARAMETERS                ###
###################################################
BASE = Path(__file__).resolve().parent
# PATH_CONF = (BASE / "../../A5_Utilis/B0_CONFIG/config_mem.json")
PATH_CONF = (BASE / "../A3_TEST/config_dram_test.json")
PATH_INIT = (BASE / "../A1_INIT")
PATH_OUT  = (BASE / "../A2_OUT")

def dram_init():

    mem_config = read_config(PATH_CONF)

    for i, item in enumerate(mem_config):
        print(f"config {i}: {item}")

if __name__ == '__main__':
    dram_init()