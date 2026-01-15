import argparse
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent
dram_name = 'dram'
PATH_INIT = (BASE / f"{dram_name}.csv")
X_WIDTH = 1024
Y_WIDTH = 1024
LUT_NUM = 8
LUT_WIDTH = 1024

def fill_token():
    print(f"Generating token data to \'{PATH_INIT}\' ...")
    with open(PATH_INIT, 'w') as f:
        for i in range(X_WIDTH):
            for j in range(Y_WIDTH):
                f.write(f"{random.randint(0, 256) - 128},")
            f.write(f"\n")

def fill_weight():
    print(f"Generating weight data to \'{PATH_INIT}\' ...")
    with open(PATH_INIT, 'w') as f:
        for i in range(X_WIDTH):
            for j in range(Y_WIDTH):
                f.write(f"{random.randint(0, LUT_WIDTH)},")
            f.write(f"\n")

def fill_zero():
    print(f"Generating zero data to \'{PATH_INIT}\' ...")
    with open(PATH_INIT, 'w') as f:
        for i in range(X_WIDTH):
            for j in range(Y_WIDTH):
                f.write(f"0,")
            f.write(f"\n")

def fill_LUT():
    
    print(f"Generating LUT data to \'{PATH_INIT}\' ...")

    LUT_WIDTH = 3 ** LUT_NUM
    X_WIDTH = 1
    while X_WIDTH < LUT_WIDTH:
        X_WIDTH = X_WIDTH * 2

    with open(PATH_INIT, 'w') as f:
        for i in range(LUT_WIDTH):
            # element = LUT_WIDTH - i - 1
            element = i
            for j in range(LUT_NUM):
                f.write(f"{element % 3},")
                element = element // 3
            f.write(f"\n")
        for i in range(LUT_WIDTH, X_WIDTH):
            for j in range(LUT_NUM):
                f.write(f"0,")
            f.write(f"\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DRAM inital dat generate') 
    parser.add_argument('--fill', default=     0, type=int)
    parser.add_argument('--XLEN', default=  1024, type=int)
    parser.add_argument('--YLEN', default=  1024, type=int)
    parser.add_argument('--LNUM', default=     8, type=int)
    parser.add_argument('--name', default='dram', type=str)

    arg = parser.parse_args()
    X_WIDTH = arg.XLEN
    Y_WIDTH = arg.YLEN
    LUT_NUM = arg.LNUM
    dram_name = arg.name
    PATH_INIT = (BASE / f"{dram_name}.csv")

    if arg.fill == 0:
        fill_zero()
    elif arg.fill == 1:
        fill_LUT()
    elif arg.fill == 2:
        fill_weight()
    elif arg.fill == 3:
        fill_token()
    # elif arg.fill == 4:
    #     X_WIDTH, Y_WIDTH = 8192, 4096;  dram_name = 'token';  fill_token
    