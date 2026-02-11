import argparse
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent
dram_name = 'dram'
PATH_INIT = (BASE / f"{dram_name}.csv")
X_WIDTH = 1024
Y_WIDTH = 1024

RANGE = 256
LUT_NUM = 8
LUT_WIDTH = 1024

TYPE = 'int'
SIGN = 0

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
                if TYPE == 'int':
                    f.write(f"{int(rand):4d}")
                else:
                    f.write(f"{rand:4.2f}")

                if j < Y_WIDTH - 1:
                    f.write(",")
                
            if i == X_WIDTH - 1: 
                f.write(f"]]\n")
            else:
                f.write(f"],\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DRAM inital dat generate') 
    parser.add_argument('--XLEN' , default=1024,  type=int)
    parser.add_argument('--YLEN' , default=1024,  type=int)
    parser.add_argument('--sign' , default=1,     type=int)
    parser.add_argument('--type' , default='int', type=str)
    parser.add_argument('--range', default=256,   type=int)
    parser.add_argument('--name', default='dram', type=str)

    arg = parser.parse_args()
    X_WIDTH = arg.XLEN
    Y_WIDTH = arg.YLEN
    TYPE  = arg.type
    SIGN  = arg.sign
    RANGE = arg.range

    dram_name = arg.name
    PATH_INIT = (BASE / f"{dram_name}.txt")

    fill()
    