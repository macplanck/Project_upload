import numpy as np

from A5_Utilis.B0_CONFIG.global_param import param
from A5_Utilis.B1_CAL.integer import LOD
from A5_Utilis.B1_CAL.fp16_parts import f16_exp
from A5_Utilis.B2_PERF.perf import PERF
from A5_Utilis.A0_BITLINEAR.bitlinear import Bitlinear

from A2_SELFATT.SELFATT import SELTATT
from A4_MEM.A0_MEM import dram,sram
from A3_MLP.MLP import proj_out, proj_upGate,  proj_down, MLP_bitwise
from A1_QKV.QKV_gen import proj_QKV

###################################################
###               MAIN Program                  ###
###################################################

if __name__ == "main":

    verbose = True

    for j in range(SELTATT.instQK.seqLen): # for every row of q
        # load a row of q from dram
        SELTATT.double_sram_toD("q", (0,j), (SELTATT.instQK.hiddenSize, 1))
        SELTATT.totalLat += 1

        for i in range (SELTATT.instQK.headNum): # for each head of q
            SELTATT.double_sram_toD("k", (SELTATT.instQK.headDim*i,0), (SELTATT.instQK.headDim, SELTATT.instQK.seqLen))
            SELTATT.one_operation(headNumb=i, rowQ=j, verbose=verbose)

        ###################################################
        ###                    MLP                      ###
        ###################################################
        #++++++++++++++++++++++++++++++++
        #++       Out Projection       ++
        #++++++++++++++++++++++++++++++++
        sram.sram_sp['out_token'], out_sum, out_max = proj_out.mem_control(SELTATT.instSramSP["qkvEle"], SELTATT.qkvSumReg, SELTATT.qkvMaxReg, dram.dram['out_weight'], sram.sram_sp['out_weight'])
        #++++++++++++++++++++++++++++++++
        #++     Gate Up projection     ++
        #++++++++++++++++++++++++++++++++
        upGate_token, upGate_sum, upGate_max = proj_upGate.mem_control(sram.sram_sp['out_token'], out_sum, out_max, [ dram.dram['up_weight'], dram.dram['gate_weight'] ], [ sram.sram_sp['up_weight'], sram.sram_sp['gate_weight'] ])
        sram.sram_sp['up_token'] = upGate_token[0]
        sram.sram_sp['gate_token'] = upGate_token[1]
        #++++++++++++++++++++++++++++++++
        #++   Bitwise Multiplication   ++
        #++++++++++++++++++++++++++++++++
        sram.sram_sp['bitwise_token'], bitwise_sum, bitwise_max = MLP_bitwise(sram.sram_sp['up_token'], sram.sram_sp['gate_token'], upGate_max[0], upGate_max[1])
        #++++++++++++++++++++++++++++++++
        #++      Down projection       ++
        #++++++++++++++++++++++++++++++++
        sram.sram_sp["token"], down_sum, down_max = proj_down.mem_control(sram.sram_sp['bitwise_token'], bitwise_sum, bitwise_max, dram.dram['down_weight'], sram.sram_sp['down_weight'])
        dram.dram['token'].load((0, j), sram.sram_sp["token"])
        ###################################################
        ###                    QKV                      ###
        ###################################################
        QKV_token, QKV_sum, QKV_max = proj_QKV.mem_control(sram.sram_sp["token"], down_sum, down_max, [ dram.dram['qkv_weightQ'], dram.dram['qkv_weightK'], dram.dram['qkv_weightV'] ], [ sram.sram_sp['qkv_weightQ'], sram.sram_sp['qkv_weightK'], sram.sram_sp['qkv_weightV'] ])
        sram.sram_sp['qkv_token_Q'] = QKV_token[0];  dram.dram['qkv_token_Q'].load((0, j), sram.sram_sp["quanQ"])
        sram.sram_sp['qkv_token_K'] = QKV_token[1];  dram.dram['qkv_token_K'].load((0, j), sram.sram_sp["quanK"])
        sram.sram_sp['qkv_token_V'] = QKV_token[2];  dram.dram['qkv_token_V'].load((0, j), sram.sram_sp["quanV"])
        
    # drain out the rest of the operation in SMD-QKV pipeline
    while (SELTATT.SMDQta != 0):
        SELTATT.SMDQKV(mode="rest", last=0, rowQ=31, verbose=verbose)

    '''
        the final outputs of SELTATTATT is returned from SMDQKV. Following variables are the variable that the output is saved at:
        1. SELTATT.instSramSP["qkvEle"] (one token at a time)
        2. SELTATT.qkvSumReg
        3. SELTATT.qkvMaxReg
        4. SELTATT.qkvSMValid (pull high for exact one cycle when it is valid)
    '''
    
    if verbose:
        print("\n---- Rest of the stages has terminate its work sucessfully ----")

    # check on final latency
    if (SELTATT.latOvHdTmp1 >= SELTATT.latOvHdTmp2):
        SELTATT.latOvHd = SELTATT.latOvHdTmp1 + SELTATT.latOvHdTmp3
    else:
        SELTATT.latOvHd = SELTATT.latOvHdTmp1 + SELTATT.latOvHdTmp2

    SELTATT.totalLat += SELTATT.latOvHd
    if verbose:
        print(f"total latency count: {SELTATT.totalLat} cycles")