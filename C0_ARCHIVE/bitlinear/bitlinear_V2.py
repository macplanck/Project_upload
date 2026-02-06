from A5_Utilis.A0_BITLINEAR.MatMul import *
from A5_Utilis.B0_CONFIG.global_param import param


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

    def sram_test(self, sram_token_in, sram_weight_in, sum_scale, max_scale):

        ############## Get Token ##############
        len_token = len(sram_token_in)
        range_token = (0, self.access_token)

        vec_token = []

        while range_token[0] < len_token:
            if range_token[1] > len_token:
                vec_receive = sram_token_in[ range_token[0] : len_token ].copy()
            else:
                vec_receive = sram_token_in[ range_token[0] : range_token[1] ].copy()
            
            vec_token += self.Bit_operation(vec_receive, sum_scale, max_scale)
            range_token = (range_token[0] + self.access_token, range_token[1] + self.access_token)
            
        # print("vec token")
        # print(vec_token)

        ############## Get Weight ##############
        len_weight = len(sram_weight_in), len(sram_weight_in[0])
        range_weight = (0, self.access_weight)
        bit_out = []
        max_out = 0
        sum_out = 0

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

            # print(f"{i:2d}: {vec_weight}")

            bit_value = self.vec_product(vec_token, vec_weight)
            bit_out.append(bit_value)

            if i == 0 or max_out < bit_value:
                max_out = bit_value
            
            sum_out += bit_value

            # print(f"sum {i:2d}: {sum_out} // {range_weight}: {vec_weight}")
            
        # print(sum_out)
        # max_out = max(sum_out)
        return bit_out, sum_out, max_out

    def Bit_operation(self, vec, sum_scale, max_scale):

        if sum_scale != None:
            vec = [ element * sum_scale for element in vec ]   # Normalization
        if max_scale != None:
            vec = [ element * max_scale for element in vec ]   # Quantization
        
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


###################################################
###                TEST Program                 ###
###################################################
if __name__ == '__main__':

    # test_bit = Bitlinear(param.hidden_size, param.intermediate_size)
    test_bit = Bitlinear(param.hidden_size, param.hidden_size)

    sram_len  = 24

    sram_token = [ i for i in range(sram_len)]
    sram_weight = [ [ j * i for j in sram_token ] for i in range(sram_len // param.LUT_NUM) ]

    sram_weight = [ [ j * i for i in range(sram_len // param.LUT_NUM) ] for j in sram_token ]

    print("---------initial token--------")
    print(sram_token)
    print("---------initial weight--------")
    for item in sram_weight:
        print(item)
    bit_out, sum_out, max_out = test_bit.sram_test(sram_token, sram_weight, None, None)

    print(f"bit")
    print(f"{bit_out}")
    print(f"{sum(bit_out)}")
    print(f"sum: {sum_out}")
    print(f"max: {max_out}")



