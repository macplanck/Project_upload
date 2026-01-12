import argparse
from pathlib import Path

BASE = Path(__file__).resolve().parent
LUT_name = 'LUT'
PATH_INIT = (BASE / f"{LUT_name}.py")


def fill_LUT():
    
    print(f"Generating LUT data to \'{PATH_INIT}\' ...")

    LUT_WIDTH = 3 ** LUT_NUM
    X_WIDTH = 1
    while X_WIDTH < LUT_WIDTH:
        X_WIDTH = X_WIDTH * 2

    with open(PATH_INIT, 'w') as f:

        f.write(f"LUT = [")

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LUT data generate') 
    parser.add_argument('--LNUM', default=    8, type=int)
    parser.add_argument('--name', default='LUT', type=str)

    arg = parser.parse_args()
    LUT_NUM = arg.LNUM
    LUT_name = arg.name
    PATH_INIT = (BASE / f"{LUT_name}.py")

    fill_LUT()
    