from A2_SELFATT.SELFATT import SELTATT
from A3_MLP import MLP
from A4_MEM.A0_MEM import dram,sram


if __name__ == "main":

    verbose = True

    for j in range(SELTATT.instQK.seqLen): # for every row of q
        # load a row of q from dram
        SELTATT.double_sram_toD("q", (0,j), (SELTATT.instQK.hiddenSize, 1))

        SELTATT.totalLat += 1

        for i in range (SELTATT.instQK.headNum): # for each head of q
            SELTATT.double_sram_toD("k", (SELTATT.instQK.headDim*i,0), (SELTATT.instQK.headDim, SELTATT.instQK.seqLen))
            SELTATT.one_operation(headNumb=i, rowQ=j, verbose=verbose)
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
# QKV gen
# SELFATT
# MLP