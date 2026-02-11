from pathlib import Path
from A5_Utilis.B0_CONFIG.read_config import read_config
from A5_Utilis.B2_PERF.perf import PERF
###################################################
###            GLOBAL PARAMETERS                ###
###################################################
BASE = Path(__file__).resolve().parent
PATH_CONF = (BASE / "../../A5_Utilis/B0_CONFIG/config_sram.json")
PATH_INIT = (BASE / "../A1_INIT")
PATH_OUT  = (BASE / "../A2_OUT")

if PERF:
    PATH_CONF = (BASE / "../../A5_Utilis/B2_PERF/config_sram.json")

###################################################
###                DRAM CLASS                   ###
###################################################
class SRAM:
    def __init__(self, config_in, name):
        self.sram_X = config_in["sram_X"]
        self.sram_Y = None
        if "sram_Y" in config_in:
            self.sram_Y = config_in["sram_Y"]
        self.name = name

###################################################
###             GLOBAL Functions                ###
###################################################
def sram_init():

    global sram_sp; sram_sp = {}
    global sram_dp; sram_dp = {}

    mem_config = read_config(PATH_CONF)

    for item in mem_config['sram_sp']:
        sram_sp[item] = SRAM(mem_config['sram_sp'][item], item)
    for item in mem_config['sram_dp']:
        sram_dp[item] = SRAM(mem_config['sram_dp'][item], item)
    
###################################################
###                MAIN CALLs                   ###
###################################################
sram_init()

###################################################
###                TEST Program                 ###
###################################################
if __name__ == '__main__':
    for item in sram_sp:
        print(f"{sram_sp[item].name} ({sram_sp[item].sram_X}, {sram_sp[item].sram_Y})")
    print(PERF)


