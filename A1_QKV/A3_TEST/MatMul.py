import numpy as np
from pathlib import Path

from A4_MEM.A0_MEM.sram import sram_sp
from A5_Utilis.B0_CONFIG.read_config import read_config
from A5_Utilis.B1_FP16.fp16_parts import f16_parts

###################################################
###            GLOBAL PARAMETERS                ###
###################################################
BASE = Path(__file__).resolve().parent
PATH_CONF = (BASE / "../A5_Utilis/B0_CONFIG/qkv_config.json")
PATH_INIT = (BASE / "../A1_INIT")
PATH_OUT  = (BASE / "../A2_OUT")

###################################################
###              TEST FUNCTIONs                 ###
###################################################
def Bit_operation(element, sum_sqrt, max):

    # Normalization
    scale = (hidden_size ** 0.5)  / sum_sqrt
    element *= scale; max *= scale

    # Quantization
    sum_sqrt 
    

def vec_product(token, weight):

    decode = sram_sp["LUT"][weight]; sum = 0

    for t, w in zip(token, decode):
        if w == 1:
            sum += t
        elif w == 2:
            sum -= t

    return sum 

###################################################
###                TEST Program                 ###
###################################################
if __name__ == '__main__':

    # read config
    bit_config = read_config

    hidden_size = bit_config["hidden_size"]

    # vec product
    vec_product([1, 1, 1, 1, 1, 1, 1, 1], 300)



