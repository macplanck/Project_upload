import argparse
import random
from pathlib import Path
from A5_Utilis.B0_CONFIG.read_config import read_config

dram_name = 'dram'

BASE = Path(__file__).resolve().parent
PATH_DRAM = (BASE / "../../A5_Utilis/B0_CONFIG/config_dram.json")
PATH_SRAM = (BASE / "../../A5_Utilis/B0_CONFIG/config_sram.json")

###################################################
###            GLOBAL PARAMETERS                ###
###################################################

X_WIDTH = 1024
Y_WIDTH = 1024

RANGE = 256
TYPE = 1
SIGN = 0

###################################################
###               FILL FUNCTIONs                ###
###################################################

def fill():

    print(f"Generating token data to \'{PATH_INIT}\' ...")

    with open(PATH_INIT, 'w') as f:

        f.write(f"{dram_name} = [")

        for i in range(X_WIDTH):
            for j in range(Y_WIDTH):

                if j == 0:
                    f.write(f"[")

                rand = random.uniform(0, RANGE)

                if SIGN:
                    rand = rand - RANGE / 2
                if TYPE:
                    f.write(f"{int(rand):4d}")
                else:
                    f.write(f"{rand:4.2f}")

                if j < Y_WIDTH - 1:
                    f.write(",")
                
            if i == X_WIDTH - 1: 
                f.write(f"]]\n")
            else:
                f.write(f"],\n")


###################################################
###               FILL DRAM                     ###
###################################################

if __name__ == '__main__':

    dram_config = read_config(PATH_DRAM)

    for item in dram_config:
        PATH_INIT = (BASE / f"dram_{item}.txt")
        X_WIDTH = dram_config[item]["dram_X"]
        Y_WIDTH = dram_config[item]["dram_Y"]
        RANGE = dram_config[item]["range"]
        SIGN = dram_config[item]["sign"] == "signed"
        TYPE = dram_config[item]["type"] == "int"
        dram_name = f"dram_{item}"
        fill()
