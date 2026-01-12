###################################################
###             BITLINEAR LUT                   ###
###################################################
import json
from pathlib import Path
from A4_MEM.A1_INIT.LUT import *

# import B01_INIT.LUT

BASE = Path(__file__).resolve().parent
PATH_CONF = (BASE / "../B00_MEM/mem_config.json")
PATH_INIT = (BASE / "../B01_INIT")

sram_sp = { "LUT": LUT }

###################################################
###                TEST Program                 ###
###################################################
if __name__ == '__main__':
    # print(LUT)
    print("Hello")



