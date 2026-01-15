import argparse
import random
from pathlib import Path
from A5_Utilis.B0_CONFIG.read_config import read_config

mem_name = 'dram'

BASE = Path(__file__).resolve().parent
PATH_INIT = BASE
PATH_DRAM = (BASE / "../../A5_Utilis/B0_CONFIG/config_dram.json")
PATH_SRAM = (BASE / "../../A5_Utilis/B0_CONFIG/config_sram.json")

###################################################
###            GLOBAL PARAMETERS                ###
###################################################

mem_type = "sram"
mem_name = "sram"

X_WIDTH = 1024
Y_WIDTH = 1024

RANGE = 256
TYPE = 1
SIGN = 0
DIV = 4

LUT_NUM = 8

###################################################
###               FILL FUNCTIONs                ###
###################################################

def fill():

    for div in range(DIV):

        print(f"Generating token data to \'{PATH_INIT}/{mem_type}_{mem_name}_V{div}.init\' ...")

        with open(f"{PATH_INIT}/{mem_type}_{mem_name}_V{div}.init", 'w') as f:
            C_WIDTH = X_WIDTH // DIV

            if div == DIV - 1:
                C_WIDTH += X_WIDTH % DIV

            if div == 0:
                f.write(f"{mem_type}[\'{mem_name}\'] = [")

            for i in range(C_WIDTH):
                for j in range(Y_WIDTH):

                    if j == 0:
                        f.write(f"[")

                    if RANGE:
                        rand = random.uniform(0, RANGE)
                    else:
                        rand = 0

                    if SIGN:
                        rand = rand - RANGE / 2
                    if TYPE:
                        f.write(f"{int(rand):4d}")
                    else:
                        f.write(f"{rand:4.2f}")

                    if j < Y_WIDTH - 1:
                        f.write(",")
                    
                if i == C_WIDTH - 1 and div == DIV - 1: 
                    f.write(f"]]\n")
                else:
                    f.write(f"],\n")

def fill_LUT():
    
    print(f"Generating LUT data to \'{PATH_INIT}/{mem_type}_{mem_name}.init\' ...")

    LUT_WIDTH = 3 ** LUT_NUM
    X_WIDTH = 1
    while X_WIDTH < LUT_WIDTH:
        X_WIDTH = X_WIDTH * 2

    with open(f"{PATH_INIT}/{mem_type}_{mem_name}.init", 'w') as f:

        f.write(f"{mem_type}[\'{mem_name}\'] = [")

        for i in range(LUT_WIDTH):
            f.write(f"["); element = i
            for j in range(LUT_NUM-1):
                f.write(f"{element % 3},")
                element = element // 3
            f.write(f"{element % 3}],\n")
        for i in range(LUT_WIDTH, X_WIDTH):
            f.write(f"[")
            for j in range(LUT_NUM-1):
                f.write(f"0,")

            if i == X_WIDTH - 1:
                f.write(f"0]]\n")
            else:
                f.write(f"0],\n")


###################################################
###               FILL DRAM                     ###
###################################################

if __name__ == '__main__':

    dram_config = read_config(PATH_DRAM)
    mem_type = "dram"

    # for item in dram_config:
    #     X_WIDTH = dram_config[item]["dram_X"]
    #     if "dram_Y" in dram_config[item]:
    #         Y_WIDTH = dram_config[item]["dram_Y"]
    #     RANGE = dram_config[item]["range"]
    #     SIGN = dram_config[item]["sign"] == "signed"
    #     TYPE = dram_config[item]["type"] == "int"
    #     mem_name = f"{item}"
    #     mem_type = "dram"
    #     fill()

    sram_config = read_config(PATH_SRAM)
    mem_type = "sram_sp"

    for item in sram_config["LUT"]:
        print(f"sram LUT: {item}")
        LUT_NUM = sram_config["LUT"][item]["LUT_num"]
        mem_name = f"{item}"
        fill_LUT()

    for item in sram_config["sram_sp"]:
        X_WIDTH = sram_config["sram_sp"][item]["sram_X"]
        if "sram_Y" in sram_config["sram_sp"][item]:
            Y_WIDTH = sram_config["sram_sp"][item]["sram_Y"]
        Y_WIDTH = sram_config["sram_sp"][item]["sram_Y"]
        RANGE = sram_config["sram_sp"][item]["range"]
        SIGN = sram_config["sram_sp"][item]["sign"] == "signed"
        TYPE = sram_config["sram_sp"][item]["type"] == "int"
        mem_name = f"{item}"
        fill()

    mem_type = "sram_dp"
    for item in sram_config["sram_dp"]:
        X_WIDTH = sram_config["sram_dp"][item]["sram_X"]
        if "sram_Y" in sram_config["sram_dp"][item]:
            Y_WIDTH = sram_config["sram_dp"][item]["sram_Y"]
        RANGE = sram_config["sram_dp"][item]["range"]
        SIGN = sram_config["sram_dp"][item]["sign"] == "signed"
        TYPE = sram_config["sram_dp"][item]["type"] == "int"
        mem_name = f"{item}"
        fill()
