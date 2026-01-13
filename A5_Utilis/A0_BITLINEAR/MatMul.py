import numpy as np
from pathlib import Path

from A4_MEM.A0_MEM.sram import sram_sp
from A5_Utilis.B1_FP16.fp16_parts import f16_exp
from A5_Utilis.B0_CONFIG.global_param import param

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
def scale(sum_sqrt, max):

    # NORM Scale
    norm_scale = (param.hidden_size ** 0.5)  / sum_sqrt

    # MAX Scale
    scale_exp, fp_kind = f16_exp(norm_scale)
    max += (param.token_size // 2 - scale_exp) 
    max_scale = 2 ** max

    return norm_scale, max_scale

def Bit_operation(element, sum_scale, max_scale):
    element *= sum_scale        # Normalization
    element *= max_scale        # Quantization
    return element

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

    # vec product
    vec_product([1, 1, 1, 1, 1, 1, 1, 1], 300)



