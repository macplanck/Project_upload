import argparse
import numpy as np
from pathlib import Path

from A4_MEM.A0_MEM.dram import dram
from A4_MEM.A0_MEM.sram import sram_sp, sram_dp, peak_sram
from A5_Utilis.B1_CAL.fp16_parts import f16_exp
from A5_Utilis.B0_CONFIG.global_param import param

###################################################
###            GLOBAL PARAMETERS                ###
###################################################
BASE = Path(__file__).resolve().parent
PATH_CONF = (BASE / "../A5_Utilis/B0_CONFIG/qkv_config.json")
PATH_INIT = (BASE / "../A1_INIT")
PATH_OUT  = (BASE / "../../A4_MEM/A2_OUT")


###################################################
###            Verify FUNCTIONs                 ###
###################################################

def MatMul_verify(wType='qkv_weightQ', tType='qkv_token', file_name='file'):

    decoded_weight = []
    wlen_X = dram[f'{wType}'].dram_X
    tlen_Y = len(dram[f'{tType}'].memory[0])

    for i in range(wlen_X):

        decoded = []

        for weight in dram[f'{wType}'].memory[i]:
            decoded += sram_sp['LUT'][weight]

        decoded_weight.append(decoded.copy())

    out_seq = []

    for t in range(tlen_Y):

        out_token = []
        for i in range(wlen_X):
            sum = 0
            for j, w in enumerate(decoded_weight[i]):
                if w == 1:
                    sum += dram[f'{tType}'].memory[j][t]
                elif w == 2:
                    sum -= dram[f'{tType}'].memory[j][t]
            
            out_token.append(sum)

        out_seq.append(out_token.copy())

    # print(f"dim: ({len(out_seq)}, {len(out_seq[0])}), ({tlen_Y, wlen_X})")

    out_seq = [ [ out_seq[j][i]  for j in range(wlen_X) ] for i in range(tlen_Y) ]

    output = {
        "out_seq": out_seq,
        "decoded_weight": decoded_weight
    }

    peak_sram(output, file_name)

    # print(f"########################################")
    # print(f"###          Verify: OUTPUT         ###")
    # print(f"########################################")
    # for i in range(wlen_X):
    #     print(out_seq[i])
    # print(f"########################################")
    # print(f"###         Verify: {wType}      ###")
    # print(f"########################################")
    # for i in range(wlen_X):
    #     print(decoded_weight[i])
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LUT data generate') 
    parser.add_argument('--w', default='qkv_weightQ', type=str)
    parser.add_argument('--t', default='qkv_token'  , type=str)
    parser.add_argument('--f', default='file'  , type=str)

    arg = parser.parse_args()
    MatMul_verify(arg.w, arg.t, arg.f)

###################################################
###              TEST FUNCTIONs                 ###
###################################################
def scale(sum_sqrt, max_scale):

    # NORM Scale
    norm_scale = (param.hidden_size ** 0.5)  / sum_sqrt

    # MAX Scale
    scale_exp, fp_kind = f16_exp(norm_scale)
    max_scale += (param.token_size // 2 - scale_exp - 1) 
    max_scale = 2 ** max_scale

    return norm_scale, max_scale

def Bit_operation(vec, sum_scale, max_scale):

    if sum_scale != None:
        vec = [ element * sum_scale for element in vec ]   # Normalization
    if max_scale != None:
        vec = [ element * max_scale for element in vec ]   # Quantization
      
    return vec.copy()

def vec_product(token, weight):

    decode = []; sum = 0

    for item in weight:
        decode += sram_sp['LUT'][item].copy()

    print(decode)

    for t, w in zip(token, decode):
        if w == 1:
            sum += t
        elif w == 2:
            sum -= t
        
    print(sum)

    return sum 

###################################################
###                TEST Program                 ###
###################################################
# if __name__ == '__main__':

    # vec product
    # vec_product([1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1, 1], [300, 300])

    # Verify function




