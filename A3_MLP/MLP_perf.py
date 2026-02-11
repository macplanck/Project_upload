import numpy as np

from A5_Utilis.B0_CONFIG.global_param import param
from A5_Utilis.B1_CAL.integer import LOD
from A5_Utilis.B1_CAL.fp16_parts import f16_exp
from A5_Utilis.B2_PERF.perf import PERF

from A5_Utilis.A0_BITLINEAR.bitlinear_perf import Bitlinear
from A4_MEM.A0_MEM.sram_perf import sram_sp, sram_dp, SRAM
from A4_MEM.A0_MEM.dram_perf import dram

###################################################
###                MAIN CALLs                   ###
###################################################
proj_out        = Bitlinear(param.hidden_size, param.hidden_size)
proj_upGate     = Bitlinear(param.hidden_size, param.intermediate_size)
proj_down       = Bitlinear(param.intermediate_size, param.hidden_size)

###################################################
###                TEST Program                 ###
###################################################
def MLP(vec_in, sum_in, max_in):

    # ********** Out projection **********
    sram_sp['out_token'], out_sum, out_max = proj_out.mem_control(vec_in, sum_in, max_in, dram['out_weight'], sram_sp['out_weight'])
    # print("****** out *******")
    print(f"max {out_max}, sum {out_sum}, {sram_sp['out_token']}")

    # ******** Up_Gate projection ********
    upGate_token, upGate_sum, upGate_max = proj_upGate.mem_control(sram_sp['out_token'], out_sum, out_max, [ dram['up_weight'], dram['gate_weight'] ], [ sram_sp['up_weight'], sram_sp['gate_weight'] ])
    sram_sp['up_token'] = upGate_token[0]
    sram_sp['gate_token'] = upGate_token[1]

    # print("****** up  *******")
    # print(f"{sram_sp['up_token']}")
    # print("****** gate ******")
    # print(f"{sram_sp['gate_token']}")

    # ****** Bitwise Multiplication ******
    range_token = (0, proj_down.num_token)
    bitwise_token = upGate_token[0]
    bitwise_sum = 0
    bitwise_max = 0

    while range_token[0] < param.intermediate_size:

        if range_token[1] > param.intermediate_size:
            range_token = (range_token[0], param.intermediate_size)

        # up_reg   = [ sram_sp['up_token'][i][0]    for i in range(range_token[0], range_token[1]) ]
        # gate_reg = [ sram_sp['gate_token'][i][0]  for i in range(range_token[0], range_token[1]) ]

        # token_seg = [ [ up_t * gate_t * (2 ** upGate_max[0]) * (2 ** upGate_max[1]) ]  for up_t, gate_t in zip(up_reg, gate_reg) ]
        # max_seg = max([  abs(t[0])    for t in token_seg  ])

        # sum_seg = sum([  t[0] * t[0]  for t in token_seg  ])

        # bitwise_token += token_seg.copy()
        # bitwise_sum += sum_seg

        # if range_token[0] == 0 or max_seg > bitwise_max:
        #     bitwise_max = max_seg

        range_token = (range_token[0] + proj_down.num_token, range_token[1] + proj_down.num_token)

    bitwise_max, _ = f16_exp(bitwise_max)
    bitwise_sum = np.float16(bitwise_sum ** 0.5)

    print(f"max {bitwise_max} sum {bitwise_sum}")

    sram_sp['bitwise_token'] = bitwise_token

    # ********** Down projection *********
    down_token, down_sum, down_max = proj_down.mem_control(sram_sp['bitwise_token'], bitwise_sum, bitwise_max, dram['down_weight'], sram_sp['down_weight'])
    return down_token, down_sum, down_max

if __name__ == '__main__':

    print("****************************************")
    print("******* This program is Executed *******")
    print("****************************************")

    if PERF:
        sram_in = sram_sp['mlp_test']
        down_token, down_sum, down_max = MLP(sram_in, 1, 1)

        print(f"down CT count: {proj_down.cycle_count}")
        print(f"gate CT count: {proj_upGate.cycle_count}")
        print(f"out  CT count: {proj_out.cycle_count}")
    else:
        print("Your should change PERF to run 32_mlp_perf")


