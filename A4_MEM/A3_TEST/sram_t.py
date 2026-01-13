###################################################
###             BITLINEAR LUT                   ###
###################################################
from pathlib import Path

from A5_Utilis.B0_CONFIG.read_config import read_config
from A4_MEM.A1_INIT.LUT import LUT

# import B01_INIT.LUT

BASE = Path(__file__).resolve().parent
PATH_CONF = (BASE / "config_sram_test.json")

sram_sp = { "LUT": LUT }
sram_dp = {}


def array_init(config_in):

    array_out = []
    
    if "sram_Y" in config_in:
        for i in range(config_in["sram_X"]):
            array = [ 0 for i in range(config_in["sram_Y"]) ]
            array_out.append(array.copy())
    else:
        array_out = [ 0 for i in range(config_in["sram_X"]) ]

    return array_out.copy()


def sram_init():

    config = read_config(PATH_CONF)
    config_sp = config["sram_sp"]
    config_dp = config["sram_dp"]

    for item in config_sp:
        sram_sp[item] = array_init(config_sp[item])
        # print(f"{item}, {sram_sp[item]}")

    for item in config_dp:
        sram_dp[item] = array_init(config_dp[item])
        print(f"{item}, {sram_dp[item]}")

###################################################
###                MAIN CALLs                   ###
###################################################
sram_init()

###################################################
###                TEST Program                 ###
###################################################
if __name__ == '__main__':
    print("Hello")





