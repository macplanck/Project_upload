import numpy as np
from A5_Utilis.A0_BITLINEAR.MatMul import *
from A5_Utilis.B0_CONFIG.global_param import param
from A5_Utilis.B1_CAL.integer import LOD
from A4_MEM.A0_MEM.dram import dram
from A4_MEM.A0_MEM.sram import peak_sram
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
        self.input_size = input_size
        self.output_size = output_size
        self.access_determine(input_size, output_size)

    def access_determine(self, input_size, output_size):

        self.available_cycles = seq_len * (param.head_dim + 1)
        self.require_accesses = input_size * output_size * param.weight_bit // param.LUT_NUM + input_size * param.token_bit

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

    def projection(self, sram_token_in, sram_weight_in, sram_token_out, sum_scale=None, max_scale=None):

        ############## Get Token ##############
        len_token = len(sram_token_in)
        range_token = (0, self.access_token)

        vec_token = []

        while range_token[0] < len_token:
            if range_token[1] > len_token:
                vec_receive = [ sram_token_in[i][0]  for i in range(range_token[0], len_token) ]
            else:
                vec_receive = [ sram_token_in[i][0]  for i in range(range_token[0], range_token[1]) ]
            vec_token += self.Bit_operation(vec_receive, sum_scale, max_scale)
            range_token = (range_token[0] + self.access_token, range_token[1] + self.access_token)
            
        # print("vec token")
        # print(vec_token)

        ############## Get Weight ##############
        len_weight = len(sram_weight_in), len(sram_weight_in[0])
        range_weight = (0, self.access_weight)
        bit_out = sram_token_out
        sum_out = 0
        max_out = 0

        for i in range(len_weight[0]):
            bit_value = 0; vec_weight = []; range_weight = (0, self.access_weight)
            while range_weight[0] < len_weight[1]:
                # print(f"Range {j:2d} {range_weight}")
                if range_weight[1] > len_weight[1]:
                    vec_receive = sram_weight_in[i][ range_weight[0] : len_weight[1] ]
                else:
                    vec_receive = sram_weight_in[i][ range_weight[0] : range_weight[1] ]
                vec_weight += vec_receive.copy()
                range_weight = (range_weight[0] + self.access_weight, range_weight[1] + self.access_weight)

            bit_value = self.vec_product(vec_token, vec_weight)
            # bit_value = bit_value + sram_token_out[i]

            bit_out[i] += bit_value
            sum_out += bit_out[i] * bit_out[i]

            if i == 0 or abs(bit_out[i]) > max_out:
                max_out = abs(bit_out[i])

            # print(f"sum {i:2d}: {sum_out} // {range_weight}: {vec_weight}")
            
        # print(sum_out)
        # max_out = max(sum_out)
        return bit_out, sum_out, max_out

    def Bit_operation(self, vec, sum_scale, max_scale):

        # print(f"prev vec: {vec}")

        if sum_scale != None:
            vec = [ element * sum_scale for element in vec ]   # Normalization
        if max_scale != None:
            vec = [ round(element * 2 ** max_scale) for element in vec ]   # Quantization

        # print(f"after vec: {vec}")
        # print(f"scale {sum_scale}, {max_scale} {sum_scale  * round(2 ** max_scale)}")
        
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
        sum_scale = (param.hidden_size ** 0.5)  / (sum_scale)

        # MAX Scale
        scale_exp, fp_kind = f16_exp(np.float16(sum_scale))
        max_scale += (param.token_bit - round(scale_exp) - 2) 

        # max_scale = 2 ** max_scale

        return sum_scale, max_scale

    def mem_control(self, vec_in, sum_in, max_in, dram_weight_in, sram_weight_in, sram_token_in):

        if not isinstance(dram_weight_in, list):
            dram_weight = [ dram_weight_in ]
            sram_weight = [ sram_weight_in ]
            sram_token  = [ sram_token_in  ]
        else:
            dram_weight = dram_weight_in
            sram_weight = sram_weight_in
            sram_token  = sram_token_in

        range_weight =     self.num_weight;   len_weight = self.input_size // 8;  pt_weight = (0, 0);  
        range_token  = (0, self.num_token);   len_token  = self.input_size

        out_token = [ [ 0 for _ in range(dram_w.dram_X) ] for dram_w in (dram_weight) ]
        out_sum = [ 0 for i in range(len(dram_weight)) ]
        out_max = [ 0 for i in range(len(dram_weight)) ]

        sum_scale, max_scale = self.scale(sum_in, max_in)
        # sum_scale, max_scale = None, None

        while pt_weight[1] < len_weight:
            if range_token[1] > len_token:
                range_token[1] = len_token
            if pt_weight[1] + range_weight > len_weight:
                range_weight = param.hidden_size - pt_weight[0]

            in_token = [ [ vec_in[i][0] ] for i in range(range_token[0], range_token[1]) ]

            for i, (dram_w, sram_w) in enumerate(zip(dram_weight, sram_weight)):
                sram_w = dram_w.load_mem(pt_weight, (self.output_size, range_weight))
                out_token[i], out_sum[i], out_max[i] = self.projection(in_token, sram_w, out_token[i], sum_scale, max_scale)

            sram_token = out_token
            pt_weight = (pt_weight[0], self.num_weight + pt_weight[1])
            range_token = (range_token[0] + self.num_token, range_token[1] + self.num_token)

        for i, (sum_t, max_t) in enumerate(zip(out_sum, out_max)):
            out_sum[i] = np.float16((sum_t ** 0.5) / (2 ** max_scale))
            max_LOD = LOD(max_t);  max_exp = param.token_bit - max_LOD - 2;  max_mem = -max_exp - max_scale;  rescale = 2 ** max_exp
            out_max[i] = max_mem

            for j, token in enumerate(out_token[i]):
                out_token[i][j] = round(token * rescale)

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

    # test_bit = Bitlinear(param.hidden_size, param.intermediate_size)
    # test_bit = Bitlinear(param.hidden_size, param.hidden_size)

    # sram_len  = 24

    # sram_input = [  i  for i in range(sram_len)]
    # sram_token = [ [i] for i in range(sram_len)]

    # sram_weight = [ [ j * i for i in range(sram_len // param.LUT_NUM) ] for j in sram_input ]

    # print("---------initial token--------")
    # print(sram_token)
    # print("---------initial weight--------")
    # for item in sram_weight:
    #     print(item)
    # bit_out, sum_out, max_out = test_bit.mem_control(sram_token, sram_weight, None, None)

    # print(f"bit")
    # print(f"{bit_out}")
    # print(f"{sum(bit_out)}")
    # print(f"sum: {sum_out}")
    # print(f"max: {max_out}")

    input_size  = 32
    output_size = 64

    test_bit = Bitlinear(input_size, output_size)
    sram_in = [ [i] for i in range(input_size) ]

    print("************************************")
    out_token, out_sum, out_max = test_bit.mem_control(sram_in, 5, 5, dram['dram_test'], [], [])
    print(f"sum {out_sum}, max {out_max}")
    print(f"max {max(out_token)}, min {min(out_token)}")
    print(out_token)
    print("************************************")

    # print("************************************")
    # out_token, out_sum, out_max = test_bit.mem_control(sram_in, 1, 1, [ dram['dram_test_1'], dram['dram_test_2'] ], [[], []], [])
    # print(f"sum {out_sum}, max {out_max}")
    # print(out_token)
    # print("************************************")

    # sram_in = [ i for i in range(input_size) ]

    # out_token = [  test_bit.vec_product(sram_in, dram['dram_test_1'].memory[i])  for i in range(output_size)  ]
    # print(out_token)

     






