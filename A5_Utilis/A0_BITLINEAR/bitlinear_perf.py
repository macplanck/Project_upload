import numpy as np
from A5_Utilis.B0_CONFIG.global_param import param
from A5_Utilis.B1_CAL.integer import LOD
from A5_Utilis.B1_CAL.fp16_parts import f16_exp

from A4_MEM.A0_MEM.dram_perf import dram
from A4_MEM.A0_MEM.sram_perf import sram_sp, sram_dp, SRAM
###################################################
###              Global Parameters              ###
###################################################
seq_len = param.seq_length
seq_len = 8192

###################################################
###           GLOBAL Callable CLASSES           ###
###################################################

class Bitlinear:

    def __init__(self, input_size, output_size):
        self.cycle_count = 0
        self.input_size = input_size
        self.output_size = output_size
        self.access_determine(input_size, output_size)

    def access_determine(self, input_size, output_size):

        self.available_cycles = seq_len * (param.head_dim + 1)
        self.require_accesses = input_size * output_size * param.weight_bit // param.LUT_NUM + input_size * param.token_bit

        print("********************************************")
        print(f"available cycles: {self.available_cycles}")
        print(f"require accesses: {self.require_accesses}")

        self.access_bits = self.require_accesses / self.available_cycles

        print(f"access bit: {self.access_bits}")

        self.access_weight = round(self.access_bits / param.weight_bit + 0.5)
        self.access_token  = round(self.access_bits / param.token_bit  + 0.5)

        self.num_token  = self.access_weight * param.LUT_NUM
        self.num_weight = self.access_weight

        print(f"access weight: {self.access_weight}")
        print(f"access  token: {self.access_token}")

        print(f"num weight: {self.num_weight}")
        print(f"num  token: {self.num_token}")

    def projection(self, sram_token_in, sram_weight_in, sram_token_out, sum_scale=None, max_scale=None):

        ########################################
        ############## Get Token ###############
        ########################################
            
        #++++++++++++++++++++++++++++++
        #+++        Changes         +++
        #++++++++++++++++++++++++++++++
        # print(f"sram in type {type(sram_token_in)}")
        # len_token = sram_token_in.sram_X
        len_token = len(sram_token_in)
        #++++++++++++++++++++++++++++++
        
        range_token = (0, self.access_token)
        vec_token = []

        while range_token[0] < len_token:
            #+++++++++++++++++++++++++++++++++++++
            #+++        Operation Code         +++
            #+++++++++++++++++++++++++++++++++++++
            range_token = (range_token[0] + self.access_token, range_token[1] + self.access_token)

        #++++++++++++++++++++++++++++++
            cycle_count  = 0
            cycle_count += 1 
        #++++++++++++++++++++++++++++++

        ########################################
        ############## Get Weight ##############
        ########################################
            
        #++++++++++++++++++++++++++++++
        #+++        Changes         +++
        #++++++++++++++++++++++++++++++
        len_weight = sram_weight_in.sram_X, sram_weight_in.sram_Y
        #++++++++++++++++++++++++++++++
        
        range_weight = (0, self.access_weight)
        bit_out = sram_token_out
        sum_out = 0
        max_out = 0

        for i in range(len_weight[0]):
            while range_weight[0] < len_weight[1]:
                #+++++++++++++++++++++++++++++++++++++
                #+++        Operation Code         +++
                #+++++++++++++++++++++++++++++++++++++
                range_weight = (range_weight[0] + self.access_weight, range_weight[1] + self.access_weight)

        #++++++++++++++++++++++++++++++
            cycle_count += 1 
            bit_out = SRAM({"sram_X": self.output_size, "sram_Y": 1 }, "proj_out")
        #++++++++++++++++++++++++++++++

        return bit_out, sum_out, max_out, cycle_count


    def Bit_operation(self, vec, sum_scale, max_scale):

        if sum_scale != None:
            vec = [ element * sum_scale for element in vec ]   # Normalization
        if max_scale != None:
            vec = [ round(element * 2 ** max_scale) for element in vec ]   # Quantization
        
        return vec.copy()

    def vec_product(self, token, weight):

        decode = []; sum = 0

        for item in weight:
            decode += sram_sp['LUT'][item].copy()

        for t, w in zip(token, decode):
            if w == 1:
                sum += t
            elif w == 2:
                sum -= t

        return sum

    def scale(self, sum_scale, max_scale):

        # NORM Scale
        sum_scale = (self.input_size ** 0.5)  / (sum_scale)

        # MAX Scale
        scale_exp, fp_kind = f16_exp(np.float16(sum_scale))
        max_scale = (param.token_bit - round(scale_exp) - max_scale - 2) 

        return sum_scale, max_scale

    def mem_control(self, vec_in, sum_in, max_in, dram_weight_in, sram_weight_in):

        if not isinstance(dram_weight_in, list):
            dram_weight = [ dram_weight_in ]
            sram_weight = [ sram_weight_in ]
            # sram_token  = [ sram_token_in  ]
        else:
            dram_weight = dram_weight_in
            sram_weight = sram_weight_in
            # sram_token  = sram_token_in

        range_weight =     self.num_weight;   len_weight = self.input_size // 8;  pt_weight = (0, 0);  
        range_token  = (0, self.num_token);   len_token  = self.input_size

        out_token = [ [ 0 for _ in range(self.output_size) ] for dram_w in (dram_weight) ]
        out_sum = [ 0 for _ in range(len(dram_weight)) ]
        out_max = [ 0 for _ in range(len(dram_weight)) ]

    #++++++++++++++++++++++++++++++
    #+++        Changes         +++
    #++++++++++++++++++++++++++++++
        sum_scale, max_scale = None, None
    #++++++++++++++++++++++++++++++

        while pt_weight[1] < len_weight:
            # print(f"range token {range_token}, {len_token}")
            if range_token[1] > len_token:
                range_token = (range_token[0], len_token)
            if pt_weight[1] + range_weight > len_weight:
                range_weight = param.hidden_size - pt_weight[0]

            #++++++++++++++++++++++++++++++
            #+++        Changes         +++
            #++++++++++++++++++++++++++++++
            in_token = [ [ 0 ] for i in range(range_token[0], range_token[1]) ]
            #++++++++++++++++++++++++++++++

            for i, (dram_w, sram_w) in enumerate(zip(dram_weight, sram_weight)):
                #print(f"dram {i} {pt_weight}, {self.output_size, range_weight}, {len(dram_w.memory), len(dram_w.memory[0])}")
                #sram_w = dram_w.load_mem(pt_weight, (self.output_size, range_weight))
                out_token[i], out_sum[i], out_max[i], cycle_count = self.projection(in_token, sram_w, out_token[i], sum_scale, max_scale)

                if i == 0:
                    self.cycle_count += cycle_count
                
            sram_sp['token_buf_1'] = out_token

            pt_weight = (pt_weight[0], self.num_weight + pt_weight[1])
            range_token = (range_token[0] + self.num_token, range_token[1] + self.num_token)

        sram_sp['token_buf_2'] = sram_sp['token_buf_1']

        for i, (sum_t, max_t) in enumerate(zip(out_sum, out_max)):
        #++++++++++++++++++++++++++++++
        #+++        Changes         +++
        #++++++++++++++++++++++++++++++
            out_sum[i] = 0
            out_max[i] = 0

            # for j, token in enumerate(out_token[i]):
            #     out_token[i][j] = [ 0 ]
        #++++++++++++++++++++++++++++++

        if isinstance(dram_weight_in, list):
            return out_token, out_sum, out_max
        else: 
            return out_token[0], out_sum[0], out_max[0]
###################################################
###          GLOBAL Callable Function           ###
###################################################


###################################################
###                TEST Program                 ###
###################################################
if __name__ == '__main__':

    input_size  = 32
    output_size = 64

    test_bit = Bitlinear(input_size, output_size)
    sram_in = [ [i] for i in range(input_size) ]

    print("************************************")
    out_token, out_sum, out_max = test_bit.mem_control(sram_in, 5, 5, dram['dram_test'], [])
    print(f"sum {out_sum}, max {out_max}")
    print(f"max {max(out_token)}, min {min(out_token)}")
    print(out_token)
    print("************************************")







