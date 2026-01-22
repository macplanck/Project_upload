import A4_MEM.A0_MEM.sram as sram
import A4_MEM.A0_MEM.dram as dram

import A2_SELFATT.QK as qk
import A2_SELFATT.QKV as qkv
import A2_SELFATT.SMD as smd
import A2_SELFATT.SMSM as smsm
import numpy as np

# execution instruction:
# PS D:\NYCU\project\EE_project\NSTC_project\designs\python_simulation\Project_upload-main> & C:\python_env\IClab\Scripts\python.exe -m A0_MAIN.main 

SEQLEN = 64
TEST   = "QKV" # QK SMSM SMD QKV
NUMHD  = 32 

if __name__ == "__main__":

    # ==== control units ====
    SMDCnt  = 0
    VColCnt = 0
    QKVCnt  = 0

    # ==== schedule simulation ==== 
    SMDQta     = 0
    NextPt     = 0
    StepQta    = 0


    # ==== latency evaluation ==== 
    totalLat = 0
    latOvHdTmp1  = 0 # latency overhead before the breakpoint
    latOvHdTmp2  = 0 # latency overhead after the breakpoint that cause data congestion
    latOvHdTmp3  = 0 # latency overhead after the breakpoint in default
    latOvHd  = 0 

    instQK     = qk.QK(seqLen=SEQLEN)
    instSMSM   = smsm.SMSM(seqLen=SEQLEN)
    instSMD    = smd.SMD(seqLen=SEQLEN)
    instQKV    = qkv.QKV(seqLen = SEQLEN)
    instSramSP = sram.sram_sp
    instDram   = dram.dram

    for i in range (instQK.headNum):
        # load q k and v from dram
        for hd in range(instQK.hiddenSize//instQK.headNum):
            instSramSP['chunkQ'][hd] = instDram['quanQ'].load_mem((instQK.hiddenSize//instQK.headNum*i + hd,0),instQK.seqLen)
            instSramSP['chunkK'][hd] = instDram['quanK'].load_mem((instQK.hiddenSize//instQK.headNum*i + hd,0),instQK.seqLen)
            instSramSP['chunkV'][hd] = instDram['quanV'].load_mem((instQK.hiddenSize//instQK.headNum*i + hd,0),instQK.seqLen)
        totalLat += 1

        for j in range(instQK.seqLen): # for every row of q
            for l in range(instQK.hiddenSize//instQK.headNum//32): # for every slice of q
                # read q_deq, q and RoPE from sram
                q = []
                ROPEsin = []
                ROPEcos = []
                for eleQ in range (32):
                    q.append(np.int8(instSramSP['chunkQ'][l*32+eleQ][j]))
                    if eleQ%2 == 0:
                        ROPEsin.append(np.float16(instSramSP['ROPE_sin'][l*16+eleQ//2][j]))
                        ROPEcos.append(np.float16(instSramSP['ROPE_cos'][l*16+eleQ//2][j]))
                deqFac = instSramSP['deqQ'][j]
                totalLat += 1
                
                # input q_deq, q
                qRslt = instQK.one_operation(np.array(q), np.int8(deqFac), np.array(ROPEcos), np.array(ROPEsin))

                assert qRslt.qFlag == 1, "q flag didn't raised as expected"

                for m in range(instQK.seqLen): # every row of k
                    # read k_deq, k and RoPE from sram
                    k = []
                    ROPEsin = []
                    ROPEcos = []
                    for eleK in range (32):
                        k.append(np.int8(instSramSP['chunkK'][l*32+eleK][m]))
                        if eleK%2 == 0:
                            ROPEsin.append(np.float16(instSramSP['ROPE_sin'][l*16+eleK//2][m]))
                            ROPEcos.append(np.float16(instSramSP['ROPE_cos'][l*16+eleK//2][m]))
                    deqFac = instSramSP['deqK'][j]
                    totalLat += 1

                    # input k_deq, k
                    kRslt = instQK.one_operation(np.array(k), np.int8(deqFac), np.array(ROPEcos), np.array(ROPEsin)) 
                    if (i == instQK.headNum - 1 and j == instQK.seqLen - 1 and m == instQK.seqLen - 1 and l == instQK.hiddenSize//instQK.headNum//32 - 1):
                        latOvHdTmp1 += instQK.lat

                    # save result
                    if (l == instQK.hiddenSize//instQK.headNum//32 - 1):
                        assert kRslt.outValid == 1, "the result of a row of score should be valid already."
                        if (TEST == "QK"):
                            print(f"progress:{kRslt.trgCnt}/{instQK.seqLen*(instQK.hiddenSize//instQK.headNum//32)*(instQK.seqLen + 1)} | value: {kRslt.prtSum}",end="\r")
                        else:
                            SMSMRslt = instSMSM.one_operation(kRslt.prtSum)
                            if (i == instQK.headNum - 1 and j == instQK.seqLen - 1):
                                latOvHdTmp1 += instSMSM.lat
                        
                        # save the value generated from SMSM into sram (SM_EXP)
                        instSramSP['smExpedWd'][m] = SMSMRslt.exped
                        if (i == instQK.headNum - 1 and j == instQK.seqLen - 1 and m == instQK.seqLen - 1):
                            latOvHdTmp1 += 1

                    if (SMDQta != 0):
                        if (SMDCnt == 0):
                            # read smExped into sram
                            instSramSP['smExpedRd'] = instDram['smExped'].load_mem((QKVCnt,0),instQK.seqLen)
                            if (i == instQK.headNum - 1 and j == instQK.seqLen - 1 and m == instQK.seqLen - 1 and l == instQK.hiddenSize//instQK.headNum//32 - 1):
                                latOvHdTmp3 += 1

                        # save a slice into smExpReg
                        smExpReg = np.array(instSramSP['smExpedRd'][SMDCnt*32:SMDCnt*32+32], dtype=np.float16)
                        smSumReg_r = instSramSP['smSum'][QKVCnt]
                        smMaxReg_r = instSramSP['smMax'][QKVCnt]
                        if (i == instQK.headNum - 1 and j == instQK.seqLen - 1 and m == instQK.seqLen - 1 and l == instQK.hiddenSize//instQK.headNum//32 - 1):
                            latOvHdTmp3 += 1

                        SMDRslt = instSMD.one_operation(smExpReg,smMaxReg_r,smSumReg_r)
                        if (i == instQK.headNum - 1 and j == instQK.seqLen - 1 and m == instQK.seqLen - 1 and l == instQK.hiddenSize//instQK.headNum//32 - 1):
                            latOvHdTmp3 += instSMD.lat

                        if (TEST == "SMD"):
                            print(f"progress: NO.{QKVCnt:2d}  {SMDCnt+1:3d}/{instQK.seqLen//32}",end="\n")
                            # print(f"j:{j}, l:{l}, m:{m}")
                        elif (TEST == "QKV"):

                            # input QK 32 elements
                            if(instQKV.QKFlag == 1):
                                instQKV.set_inputs(inNum=SMDRslt, dqFac = np.zeros(32)) # current dqFac input doesn't affect the result
                                QKVRslt, VColCnt = instQKV.one_operation()
                                
                            else:
                                instQKV.set_inputs(inNum=instSramSP['chunkV'][instQKV.colCnt][SMDCnt*32:SMDCnt*32+32], dqFac=instSramSP['deqV'][SMDCnt*32:SMDCnt*32+32]) # current dqFac input doesn't affect the result
                                QKVRslt, VColCnt = instQKV.one_operation()
                                print(f"progress: QKVrow.{QKVCnt:5d}, seq slice.{SMDCnt:5d}, input with V at NO.{VColCnt+1:3d}/{instQK.hiddenSize//instQK.headNum}",end="\r")

                            if (i == instQK.headNum - 1 and j == instQK.seqLen - 1 and m == instQK.seqLen - 1 and l == instQK.hiddenSize//instQK.headNum//32 - 1):
                                latOvHdTmp3 += instQKV.lat

                        # advance SMDCnt
                        if (VColCnt == (instQK.hiddenSize // instQK.headNum - 1) or TEST == "SMD"):
                            if (SMDCnt == (instQK.seqLen//32)-1):
                                SMDQta -= 1 # if one row of the element are processed, subtract one row of quota
                                SMDCnt = 0
                                if (QKVCnt != instQK.seqLen - 1):
                                    QKVCnt += 1
                                else:
                                    print(f"\n---- done chunk {i} ----")
                                    QKVCnt = 0
                            else:
                                SMDCnt +=1
                
                # write smExped into dram
                instDram['smExped'].store_mem((j,0),instSramSP['smExpedWd'])
                if (i == instQK.headNum - 1 and j == instQK.seqLen - 1 and m == instQK.seqLen - 1 and l == instQK.hiddenSize//instQK.headNum//32 - 1):
                    latOvHdTmp3 += 1

            # a row of softmax finding is done
            if (TEST == "SMSM"):
                assert SMSMRslt.DmnValid == 1, "the summation and max value should be valid already."
                print(f"progress:{SMSMRslt.trgCnt//instSMSM.seqLen:2d}/{instSMSM.seqLen} | Dnm: {SMSMRslt.sumDnm:10.6f} | maxExp: {SMSMRslt.maxExp}",end="\n")
            elif (TEST != "QK"):
                # save into regs (FP16*1 and INT8*1) (SM_SUM and SM_MAX)
                smSumReg_w = SMSMRslt.sumDnm
                smMaxReg_w = SMSMRslt.maxExp

                instSramSP['smSum'][j] = smSumReg_w
                instSramSP['smMax'][j] = smMaxReg_w

                # enable the SMD evaluation
                SMDQta += 1

    # drain out the rest of the operation in SMD-QKV pipeline
    while (SMDQta != 0):
        if (SMDCnt == 0):
            # read smExped into sram
            instSramSP['smExpedRd'] = instDram['smExped'].load_mem((QKVCnt,0),instQK.seqLen)
            latOvHdTmp2 += 1

        # save first slice of into smExpReg
        smExpReg = np.array(instSramSP['smExpedRd'][SMDCnt*32:SMDCnt*32+32], dtype=np.float16)

        smSumReg_r = instSramSP['smSum'][QKVCnt]
        smMaxReg_r = instSramSP['smMax'][QKVCnt]
        latOvHdTmp2 += 1

        SMDRslt = instSMD.one_operation(smExpReg,smMaxReg_r,smSumReg_r)
        if (SMDQta == 1 and SMDCnt == (instQK.seqLen//32)-1 and QKVCnt == instQK.seqLen - 1):
            latOvHdTmp2 += instSMD.lat
        else:
            latOvHdTmp2 += 1

        if (TEST == "SMD"):
            print(f"progress: NO.{QKVCnt:2d}  {SMDCnt+1:3d}/{instQK.seqLen//32}",end="\n")
            # print(f"j:{j}, l:{l}, m:{m}")
        elif (TEST == "QKV"):

            # input QK 32 elements
            if(instQKV.QKFlag == 1):
                instQKV.set_inputs(inNum=SMDRslt, dqFac = np.zeros(32)) # current dqFac input doesn't affect the result
                QKVRslt, VColCnt = instQKV.one_operation()
            else:
                instQKV.set_inputs(inNum=instSramSP['chunkV'][instQKV.colCnt][SMDCnt*32:SMDCnt*32+32], dqFac=instSramSP['deqV'][SMDCnt*32:SMDCnt*32+32]) # current dqFac input doesn't affect the result
                QKVRslt, VColCnt = instQKV.one_operation()
                print(f"progress: QKVrow.{QKVCnt:5d}, seq slice.{SMDCnt:5d}, input with V at NO.{VColCnt+1:3d}/{instQK.hiddenSize//instQK.headNum}",end="\r")
                # print(f"j:{j}, l:{l}, m:{m}")

            if (SMDQta == 1 and SMDCnt == (instQK.seqLen//32)-1 and QKVCnt == instQK.seqLen - 1):
                latOvHdTmp2 += instSMD.lat
            else:
                latOvHdTmp2 += 1

        # advance SMDCnt
        if (VColCnt == (instQK.hiddenSize // instQK.headNum - 1) or TEST == "SMD"):
            if (SMDCnt == (instQK.seqLen//32)-1):
                SMDQta -= 1 # if one row of the element are processed, subtract one row of quota
                SMDCnt = 0
                if (QKVCnt != instQK.seqLen - 1):
                    QKVCnt += 1
                else:
                    QKVCnt = 0
            else:
                SMDCnt +=1
        
    print("\n---- Rest of the stages has terminate its work sucessfully ----")

    # check on final latency
    if (latOvHdTmp1 >= latOvHdTmp2):
        latOvHd = latOvHdTmp1 + latOvHdTmp3
    else:
        latOvHd = latOvHdTmp1 + latOvHdTmp2

    totalLat += latOvHd
    print(f"total latency count: {totalLat} cycles")

