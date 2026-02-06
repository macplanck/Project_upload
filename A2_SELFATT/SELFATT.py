import A4_MEM.A0_MEM.sram as sram
import A4_MEM.A0_MEM.dram as dram

import QKVSM as qkvsm
import QK as qk
import QKV as qkv
import SMD as smd
import SMSM as smsm
import numpy as np

# execution instruction:
# PS D:\NYCU\project\EE_project\NSTC_project\designs\python_simulation\Project_upload-main> & C:\python_env\IClab\Scripts\python.exe -m A0_MAIN.main 

SEQLEN = 64
TEST   = "QKV" # QK SMSM SMD QKV
NUMHD  = 32 

class SELTATT ():
    def __init__(self, seqlen = SEQLEN, test = TEST, numhd = NUMHD):
        # ==== settings ====
        self.seqlen = seqlen
        self.test   = test
        self.numhd  = numhd

        # ==== control units ====
        self.SMDCnt   = 0
        self.VColCnt  = 0
        self.QKCnt    = 0
        self.tkCnt    = 0
        self.qBufPter = 0
        self.kBufPter = 0
        self.vBufPter = 0
        self.smSumPter = 0
        self.smMaxPter = 0

        # ==== data storing registers ====
        self.smSumReg_w = 0
        self.smMaxReg_w = 0
        self.smExpReg   = np.zeros(32)
        self.smSumReg_r = 0
        self.smMaxReg_r = 0
        self.qkvSumReg  = 0
        self.qkvMaxReg  = 0


        # ==== unroll variables ====
        self.SMDQta  = 0

        # ==== latency evaluation ====
        self.totalLat     = 0
        self.latOvHdTmp1  = 0 # latency overhead before the breakpoint
        self.latOvHdTmp2  = 0 # latency overhead after the breakpoint that cause data congestion
        self.latOvHdTmp3  = 0 # latency overhead after the breakpoint in default
        self.latOvHd      = 0 

        # ==== instantiated functions ====
        self.instQK     = qk.QK(seqLen=seqlen)
        self.instSMSM   = smsm.SMSM(seqLen=seqlen)
        self.instSMD    = smd.SMD(seqLen=seqlen)
        self.instQKV    = qkv.QKV(seqLen=seqlen)
        self.instQKVSM  = qkvsm.QKVSM(seqLen=seqlen)
        self.instSramSP = sram.sram_sp
        self.instDram   = dram.dram

    def operations (self, verbose = True):
        '''
        Docstring for operations

        this session operates the entire process from QK to QKV.

        :param verbose: bool, it determine whether the process should be printed or not.
        '''
        for j in range(self.instQK.seqLen): # for every row of q
            # load a row of q from dram
            self.double_sram_toD("q", (0,j), (self.instQK.hiddenSize, 1))

            self.totalLat += 1

            for i in range (self.instQK.headNum): # for each head of q
                self.double_sram_toD("k", (self.instQK.headDim*i,0), (self.instQK.headDim, self.instQK.seqLen))
                self.one_operation(headNumb=i, rowQ=j, verbose=verbose)
        # drain out the rest of the operation in SMD-QKV pipeline
        while (self.SMDQta != 0):
            self.SMDQKV(mode="rest", last=0, rowQ=31, verbose=verbose)
        
        if verbose:
            print("\n---- Rest of the stages has terminate its work sucessfully ----")

        # check on final latency
        if (self.latOvHdTmp1 >= self.latOvHdTmp2):
            self.latOvHd = self.latOvHdTmp1 + self.latOvHdTmp3
        else:
            self.latOvHd = self.latOvHdTmp1 + self.latOvHdTmp2

        self.totalLat += self.latOvHd
        if verbose:
            print(f"total latency count: {self.totalLat} cycles")

    def one_operation(self, headNumb, rowQ, verbose = True):
        '''
        Docstring for one_operation:
        output one token, a sum and a max for every call.
        
        :param headNumb: int, the headNum index it is processing.
        :param rowQ: int, it indicates the row of q that it is processing.
        :param verbose: bool, This parameter pass down the permission for the printing function to show up on the terminal.
        '''
        for l in range(self.instQK.headDim//32): # for every slice of q chunk
            # read q_deq, q and RoPE from sram
            q = []
            ROPEsin = []
            ROPEcos = []
            for eleQ in range (32):
                qSramKey = self.double_sram_RS("q")
                q.append(np.int8(self.instSramSP[qSramKey][l*32+eleQ][0]))
                if eleQ%2 == 0:
                    ROPEsin.append(np.float16(self.instSramSP['ROPE_sin'][l*16+eleQ//2][rowQ]))
                    ROPEcos.append(np.float16(self.instSramSP['ROPE_cos'][l*16+eleQ//2][rowQ]))
            deqFac = self.instSramSP['deqQ'][rowQ]
            self.totalLat += 1
            
            # input q_deq, q
            qRslt = self.instQK.one_operation(np.array(q), np.int8(deqFac), np.array(ROPEcos), np.array(ROPEsin))

            assert qRslt.qFlag == 1, "q flag didn't raised as expected"

            for m in range(self.instQK.seqLen): # every row of partial vector from k

                lastDur = headNumb == self.instQK.headNum - 1 and rowQ == self.instQK.seqLen - 1 and m == self.instQK.seqLen - 1 and l == self.instQK.hiddenSize//32 - 1

                # read k_deq, k and RoPE from sram
                k = []
                ROPEsin = []
                ROPEcos = []
                for eleK in range (32):
                    kSramKey = self.double_sram_RS("k")
                    k.append(np.int8(self.instSramSP[kSramKey][l*32+eleK][m]))
                    if eleK%2 == 0:
                        ROPEsin.append(np.float16(self.instSramSP['ROPE_sin'][l*16+eleK//2][m]))
                        ROPEcos.append(np.float16(self.instSramSP['ROPE_cos'][l*16+eleK//2][m]))
                deqFac = self.instSramSP['deqK'][m]
                self.totalLat += 1

                # input k_deq, k
                kRslt = self.instQK.one_operation(np.array(k), np.int8(deqFac), np.array(ROPEcos), np.array(ROPEsin)) 
                if (lastDur):
                    self.latOvHdTmp1 += self.instQK.lat # checking

                # save result
                if (l == self.instQK.headDim//32 - 1): # the last slice of q
                    assert kRslt.outValid == 1, "the result of a row of score should be valid already."
                    if (self.test == "QK"):
                        if verbose:
                            print(f"progress:{kRslt.trgCnt}/{self.instQK.seqLen*self.instQK.headDim*(self.instQK.seqLen + 1)} | value: {kRslt.prtSum}",end="\r")
                    else:
                        SMSMRslt = self.instSMSM.one_operation(kRslt.prtSum)
                        if (headNumb == self.instQK.headNum - 1 and rowQ == self.instQK.seqLen - 1):
                            self.latOvHdTmp1 += self.instSMSM.lat
                    
                        # save the value generated from SMSM into sram (SM_EXP)
                        self.instSramSP['smExpedWd'][m] = SMSMRslt.exped
                        if (headNumb == self.instQK.headNum - 1 and rowQ == self.instQK.seqLen - 1 and m == self.instQK.seqLen - 1):
                            self.latOvHdTmp1 += 1

                if (self.SMDQta != 0):
                    self.SMDQKV(mode="during", last=lastDur, rowQ=rowQ, verbose=verbose)
            
            # write smExped into dram
            self.instDram['smExped'].store_mem((0,headNumb),np.array(self.instSramSP['smExpedWd']).reshape(-1,1).tolist()) # we store for a token only, since the fetching rate at the later stage is faster by (headDim-seqLen)/32 + 2 cycles
            if (lastDur):
                self.latOvHdTmp3 += 1

        # a row of softmax finding is done
        if (self.test == "SMSM"):
            assert SMSMRslt.DmnValid == 1, "the summation and max value should be valid already."
            if (verbose):
                print(f"progress:{SMSMRslt.trgCnt//self.instSMSM.seqLen:2d}/{self.instQK.seqLen*self.instQK.headNum} | Dnm: {SMSMRslt.sumDnm:10.6f} | maxExp: {SMSMRslt.maxExp}",end="\n")
        elif (self.test != "QK"):
            # save into regs (FP16*1 and INT8*1) (SM_SUM and SM_MAX)
            self.smSumReg_w = SMSMRslt.sumDnm
            self.smMaxReg_w = SMSMRslt.maxExp
            
            smSumSramKey = self.double_sram_RS('smSum', smCnt = headNumb, act = 'toS')
            smMaxSramKey = self.double_sram_RS('smMax', smCnt = headNumb, act = 'toS')
            self.instSramSP[smSumSramKey][headNumb] = self.smSumReg_w
            self.instSramSP[smMaxSramKey][headNumb] = self.smMaxReg_w

            # enable the SMD evaluation
            self.SMDQta += 1
    
    def double_sram_RS (self, name, smCnt = 0, act = "toR"):
        '''
        Docstring for double_sram_RS
        
        :param name:  buffer name
        :param smCnt: it is only referenced in act = "toS" and indicates the when to toggle for the sm pointers
        :param act:   either "toR" (register) or "toS" (SRAM)
        '''
        smSize  = self.instQK.headNum
        SramKey = None


        if act == "toR":

            match name:
                case "q":
                    SramKey = 'tkQ' + str(int(not bool(self.qBufPter)))

                case "k":
                    SramKey = 'tkK' + str(int(not bool(self.kBufPter)))

                case "v":
                    SramKey = 'tkV' + str(int(not bool(self.vBufPter)))
                
                case "smSum":
                    SramKey = 'smSum' + str(int(not bool(self.smSumPter)))

                case "smMax":
                    SramKey = 'smMax' + str(int(not bool(self.smMaxPter)))

            if SramKey == None:
                raise NameError(f"there are no double buffer named {name}")
            
        elif act == "toS":

            match name:
                case "smSum":
                    SramKey = 'smSum' + str(int(bool(self.smSumPter)))
                    if (smCnt == smSize-1):
                        self.smSumPter ^= 1

                case "smMax":
                    SramKey = 'smMax' + str(int(bool(self.smMaxPter)))
                    if (smCnt == smSize-1):
                        self.smMaxPter ^= 1

        else:
            raise NameError(f"Vague instruction for act. {act} is detected. It should be either 'toR' or 'toS'.")
        
        if SramKey == None:
            raise NameError(f"there are no double buffer named {name}")
        
        return SramKey

        
    
    def double_sram_toD (self, name, dramStPt, dramRng):
        '''
        Docstring for double_sram_toD
        p.s. the number of the memory should be arbitrated by internal control
        :param name:     string, it could be either q, k or v.
        :param dramStPt: tuple, this parameter refers to the start point for target dram.
        :param dramRng:  tuple or integer, it indicate the range that the dram value is copy from.
        '''

        assert name == "q" or name == "k" or name == "v", f" {name} isn't a valid name for the double buffer"

        match name:
            case "q":
                SramKey = 'tkQ' + str(self.qBufPter)
                DramKey = 'quanQ'
                self.qBufPter ^= 1

            case "k":
                SramKey = 'tkK' + str(self.kBufPter)
                DramKey = 'quanK'
                self.kBufPter ^= 1

            case "v":
                SramKey = 'tkV' + str(self.vBufPter)
                DramKey = 'quanV'
                self.vBufPter ^= 1

        self.instSramSP[SramKey] = self.instDram[DramKey].load_mem(dramStPt, dramRng)
            
            

    def SMDQKV (self, mode, last, rowQ, verbose = True):
        # the helper function that help oerations looks clearer
        '''
        Docstring for SMDQKV
        
        :param mode: string, could be either "rest" or "during", the former one refers to processing the rest of the information, whereas the latter one means the function is called when the first pipeline (QK to SMSM) are still streaming
        :param last: bool, it is only used in `mode` == "during", which refers to the last processing when the previous pipeline is still in use.
        :param rowQ: integer, it is only used in `mode` == "during", which refers to the q row that it is processing.
        :param verbose: bool, the parameter decide whether the progress will appear in the terminal or not
        '''
        assert mode == "rest" or mode == "during", "the mode should be either rest or during"

        mdDuringTrig = last and mode == "during"

        if (self.SMDCnt == 0):
            # read smExped into sram
            self.instSramSP['smExpedRd'] = self.instDram['smExped'].load_mem((0,self.tkCnt),(self.instQK.seqLen,1))
            self.double_sram_toD("v", (self.instQK.headDim*self.tkCnt,0), (self.instQK.headDim, self.instQK.seqLen))
            if mode == "rest":
                self.latOvHdTmp2 += 1
            elif (mdDuringTrig):
                self.latOvHdTmp3 += 1

        # save first slice of into smExpReg
        self.smExpReg = np.array(self.instSramSP['smExpedRd'][self.SMDCnt*32:self.SMDCnt*32+32], dtype=np.float16).flatten()

        smSumSramKey= self.double_sram_RS (name = 'smSum', smCnt = 0, act = "toR")
        smMaxSramKey= self.double_sram_RS (name = 'smMax', smCnt = 0, act = "toR")
        self.smSumReg_r = self.instSramSP[smSumSramKey][self.tkCnt] 
        self.smMaxReg_r = self.instSramSP[smMaxSramKey][self.tkCnt]
        if mode == "rest":
            self.latOvHdTmp2 += 1
        elif (mdDuringTrig):
            self.latOvHdTmp3 += 1

        SMDRslt = self.instSMD.one_operation(self.smExpReg,self.smMaxReg_r,self.smSumReg_r)
        if mode == "rest":
            if (self.SMDQta == 1 and self.SMDCnt == (self.instQK.seqLen//32)-1 and self.QKCnt == self.instQK.seqLen - 1):
                self.latOvHdTmp2 += self.instSMD.lat
            else:
                self.latOvHdTmp2 += 1
        elif (mdDuringTrig):
            self.latOvHdTmp3 += self.instSMD.lat

        if (self.test == "SMD" and verbose):
            print(f"progress: NO.{self.QKCnt:2d}  {self.SMDCnt+1:3d}/{self.instQK.seqLen//32}",end="\r")
            # print(f"j:{j}, l:{l}, m:{m}")
        elif (self.test == "QKV"):

            # input QK 32 elements
            if(self.instQKV.QKFlag == 1):
                self.instQKV.set_inputs(inNum=SMDRslt, dqFac = np.zeros(32)) # current dqFac input doesn't affect the result
                QKVRslt, self.VColCnt = self.instQKV.one_operation()
            else:
                vSramKey = self.double_sram_RS("v")
                self.instQKV.set_inputs(inNum=np.asarray(self.instSramSP[vSramKey][self.instQKV.colCnt][self.SMDCnt*32:self.SMDCnt*32+32]).flatten(), dqFac=self.instSramSP['deqV'][self.SMDCnt*32:self.SMDCnt*32+32]) # current dqFac input doesn't affect the result
                QKVRslt, self.VColCnt = self.instQKV.one_operation()
                
                
                if (self.instQKV.outValid):
                    # store QKVRslt into SRAM (one element at a time)
                    self.instSramSP["qkvEle"][self.tkCnt*32 + self.VColCnt] = QKVRslt

                    # sum max searching (one sum of square and max per token)
                    self.qkvSumReg,self.qkvMaxReg,qkvSMValid = self.instQKVSM.one_operation(inSum = QKVRslt, theSum = self.qkvSumReg, theMax = self.qkvMaxReg, colMarker = self.VColCnt + self.tkCnt*self.instQK.headDim)

                    if (qkvSMValid):
                        print(f"output token No.{self.QKCnt}, with the sum value {self.qkvSumReg} and max value {self.qkvMaxReg}")

                if verbose:
                    print(f"progress: QKVrow.{self.QKCnt:5d}, head.{self.tkCnt:5d}, seq slice.{self.SMDCnt:5d}, input with V at NO.{self.VColCnt:3d}/{self.instQK.hiddenSize//self.instQK.headNum-1}",end="\r")
                # print(f"j:{j}, l:{l}, m:{m}")
            
            
            if mode == "rest":
                if (self.SMDQta == 1 and self.SMDCnt == (self.instQK.seqLen//32)-1 and self.QKCnt == self.instQK.seqLen - 1):
                        self.latOvHdTmp2 += self.instQKV.lat
                        self.latOvHdTmp2 += self.instQKVSM.lat
                        self.latOvHdTmp2 += 1 # write back t0 qkvSum and qkvMax SRAM
                else:
                    self.latOvHdTmp2 += 1
            elif (mdDuringTrig):
                self.latOvHdTmp3 += self.instQKV.lat
                self.latOvHdTmp3 += self.instQKVSM.lat
                self.latOvHdTmp3 += 1 # write back t0 qkvSum and qkvMax SRAM

        # advance self.SMDCnt
        if (self.VColCnt == (self.instQK.hiddenSize // self.instQK.headNum - 1) or self.test == "SMD"):
            if (self.SMDCnt == (self.instQK.seqLen//32)-1):
                self.SMDQta -= 1 # if one row of the element are processed, subtract one row of quota
                self.SMDCnt = 0

                if (self.tkCnt != self.instQK.headNum - 1):
                    self.tkCnt += 1
                else:
                    self.tkCnt = 0
                    if (mode == "during" and verbose):
                        print(f"\n---- done row {self.QKCnt} ----")

                    if (self.QKCnt != self.instQK.seqLen - 1):
                        self.QKCnt += 1
                    else:
                        self.QKCnt = 0
            else:
                self.SMDCnt +=1

if __name__ == "__main__":
    instSELTATT = SELTATT()
    instSELTATT.operations()

