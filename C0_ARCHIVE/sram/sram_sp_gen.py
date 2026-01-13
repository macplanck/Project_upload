import argparse
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent
sram_name = 'sram'
PATH_INIT = (BASE / f"{sram_name}_sp_init.txt")
ADDR_WIDTH = 1024
LUT_WIDTH  = 1024
LUT_NUM    = 8

def fill_token():
    with open(PATH_INIT, 'w') as f:
        f.write(f"{ADDR_WIDTH}\n")
        for i in range(ADDR_WIDTH):
            f.write(f"{random.randint(0, 256) - 128}\n")

def fill_weight():
    with open(PATH_INIT, 'w') as f:
        f.write(f"{LUT_WIDTH}\n")
        for i in range(LUT_WIDTH):
            f.write(f"{random.randint(0, LUT_WIDTH)}\n")

def fill_zero():
    with open(PATH_INIT, 'w') as f:
        f.write(f"{ADDR_WIDTH}\n")
        for i in range(ADDR_WIDTH):
            f.write(f"0\n")

def fill_LUT():
    # with open(PATH_INIT, 'w') as f:
    #     for i in range(LUT_WIDTH):
    #         # element = LUT_WIDTH - i - 1
    #         element = 1
    #         for j in range(LUT_NUM):
    #             f.write(f"{element % 3} ")
    #             element = element // 3
    #         f.write(f"\n")
    #     for i in range(LUT_WIDTH, ADDR_WIDTH):
    #         for j in range(LUT_NUM):
    #             f.write(f"0 ")
    #         f.write(f"\n")

    with open(PATH_INIT, 'w') as f:

        f.write(f" {ADDR_WIDTH}\n")

        for i in range(LUT_WIDTH):
            element = i 
            value = 0
            for j in range(LUT_NUM):
                value = value * 4 + (element % 3)
                element = element // 3
            f.write(f" {value}\n")

        for i in range(LUT_WIDTH, ADDR_WIDTH):
            f.write(f"0\n")

    # print(f"addr bit: {np.log2(ADDR_WIDTH)}")
    # print(f"avg  bit: {np.log2(ADDR_WIDTH) * 4096 / LUT_NUM / 4096}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SRAM_single_port inital txt generate') 
    parser.add_argument('--fill', default=     0, type=int)
    parser.add_argument('--ADDR', default=  1024, type=int)
    parser.add_argument('--LNUM', default=     8, type=int)
    parser.add_argument('--name', default='sram', type=str)

    arg = parser.parse_args()
    ADDR_WIDTH = arg.ADDR
    LUT_NUM = arg.LNUM
    sram_name = arg.name
    PATH_INIT = (BASE / f"{sram_name}_sp_init.txt")

    if arg.fill == 0 or arg.fill == 2:
        LUT_WIDTH = 3 ** LUT_NUM
        ADDR_WIDTH = 1
        while ADDR_WIDTH < LUT_WIDTH:
            ADDR_WIDTH = ADDR_WIDTH * 2

    if arg.fill == 0:
        print(f"Generating LUT data to \'{sram_name}_sp_init.txt\' ...")
        fill_LUT()
    elif arg.fill == 1:
        print(f"Generating zero data to \'{sram_name}_sp_init.txt\' ...")
        fill_zero()
    elif arg.fill == 2:
        print(f"Generating weight data to \'{sram_name}_sp_init.txt\' ...")
        fill_weight()
    elif arg.fill == 3:
        print(f"Generating token data to \'{sram_name}_sp_init.txt\' ...")
        fill_token()
    