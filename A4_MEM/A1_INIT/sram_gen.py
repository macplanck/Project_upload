import argparse
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent

rng = np.random.default_rng()#seed=42

def random_int8(rng = rng, low: int = -128, high: int = 127) -> np.int8:
    assert -128 <= low <= high <= 127
    return np.int8(rng.integers(low, high + 1))

def fmt_fp16(x: np.float16, decimals: int = 8) -> str:
    return f"{float(x):.{decimals}f}"

def random_fp16_uniform(
    rng = rng,
    low: float = -65504.0,
    high: float = 65504.0,
) -> np.float16:
    assert low < high
    return np.float16(rng.uniform(low, high))

def gen_row(length, dtype = "int"):
    
    lst = "["
    for l in range(length):
        if (dtype == "int"):
            num = str(random_int8(low= -5, high = 5))
        else:
            # dtype == "fp16"
            num = fmt_fp16(random_fp16_uniform(low= -1.0, high = 1.0))
        
        if num[0] != "-":
            addSpace = " "
        else:
            addSpace = ""

        lst = lst + addSpace + num

        if (l == length-1):
            lst += "]"
        else:
            lst += ", "

    return lst

def fill_SRAM():
    global SRAM_dtype, SRAM_name, SRAM_x, SRAM_y

    # sanity check:
    if (SRAM_dtype != 'float' and SRAM_dtype != 'int'):
        raise ValueError(f"specified data type should be either float ar integer, got '{SRAM_dtype}' instead.")
    
    print(f"Generating LUT data to \'{PATH_INIT}\' ...")


    with open(PATH_INIT, 'a') as f:
        if SRAM_y != 1:
            f.write(f"\n{SRAM_name} = [\n")
            for i in range(SRAM_x):
                lst = gen_row(SRAM_y, dtype=SRAM_dtype)
                f.write(f"{lst}")
                if (i == SRAM_x-1):
                    f.write("\n]\n")
                else:
                    f.write(", \n")
        else:
            f.write(f"\n{SRAM_name} = ")
            lst = gen_row(SRAM_x, dtype=SRAM_dtype)
            f.write(f"{lst}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sram generate') 
    parser.add_argument('--x',    default= 4096,   type=int, help="the first dimension of the sram list")
    parser.add_argument('--y',    default= 1,  type=int, help="the second dimension of hte sram list")
    parser.add_argument('--name', default= 'V_deq',  type=str)
    parser.add_argument('--dtype', default= 'int',  type=str)

    arg = parser.parse_args()
    SRAM_x = arg.x
    SRAM_y = arg.y
    SRAM_name = arg.name
    SRAM_dtype = arg.dtype
    PATH_INIT = (BASE / f"sram.py")

    fill_SRAM()