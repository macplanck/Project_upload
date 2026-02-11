import numpy as np
from A5_Utilis.A0_BITLINEAR.bitlinear import Bitlinear
from A5_Utilis.B0_CONFIG.global_param import param
from A5_Utilis.B1_CAL.integer import LOD
from A4_MEM.A0_MEM.sram import sram_dp, sram_sp, peak_sram
from A4_MEM.A0_MEM.dram import dram
from A5_Utilis.B2_PERF.perf import PERF
###################################################
###                MAIN CALLs                   ###
###################################################
proj_QKV = Bitlinear(param.hidden_size, param.hidden_size)

###################################################
###                TEST Program                 ###
###################################################
def QKV_gen(vec_in, sum_in, max_in):
    
    QKV_token, QKV_sum, QKV_max = proj_QKV.mem_control(vec_in, sum_in, max_in, [ dram['qkv_weightQ'], dram['qkv_weightK'], dram['qkv_weightV'] ], [ sram_sp['qkv_weightQ'], sram_sp['qkv_weightK'], sram_sp['qkv_weightV'] ])
    print(f"** 1 ** {QKV_token[0]}")
    print(f"** 2 ** {QKV_token[1]}")
    print(f"** 3 ** {QKV_token[2]}")

if __name__ == '__main__':

    print("****************************************")
    print("******* This program is Executed *******")
    print("****************************************")

    if PERF:
        print("Your should change PERF to run 11_QKV_gen")
    else:
        sram_in = [ [i]  for i in range(param.hidden_size)]
        QKV_gen(sram_in, 1, 1)
    # QKV_verify()

