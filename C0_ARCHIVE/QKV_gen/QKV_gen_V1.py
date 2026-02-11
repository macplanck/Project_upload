from A5_Utilis.A0_BITLINEAR.bitlinear import Bitlinear
from A5_Utilis.B0_CONFIG.global_param import param
from A4_MEM.A0_MEM.sram import sram_dp, sram_sp, peak_sram

from A4_MEM.A0_MEM.dram import dram
###################################################
###                MAIN CALLs                   ###
###################################################
bitlinear = Bitlinear(param.hidden_size, param.hidden_size)

###################################################
###                TEST Program                 ###
###################################################
def QKV_gen():
    
    pt_weight = (0, 0);  range_weight = bitlinear.num_weight;  len_weight = param.hidden_size // 8
    pt_token  = (0, 0);  range_token  = bitlinear.num_token;   len_token  = param.hidden_size

    out_seq = []
    out_sum = []
    out_max = []

    # print(f"########################################")
    # print(f"###          TEST: OUTPUT         ###")
    # print(f"########################################")

    for i in range(param.seq_length):
        pt_weight = (0, 0);  range_weight = bitlinear.num_weight;  len_weight = param.hidden_size // 8
        pt_token  = (0, i);  range_token  = bitlinear.num_token;   len_token  = param.hidden_size

        while pt_weight[1] < len_weight:
        
            ################ [LOAD SRAM] ################
            if pt_token[0]  + range_token  > len_token:
                range_token = param.hidden_size - pt_token[0]
            if pt_weight[1] + range_weight > len_weight:
                range_weight = param.hidden_size - pt_weight[0]

            sram_sp['qkv_token'  ] = dram['qkv_token'  ].load_mem(pt_token , (range_token, 1))
            sram_sp['qkv_weightQ'] = dram['qkv_weightQ'].load_mem(pt_weight, (param.hidden_size, range_weight))
            # sram_sp['qkv_weightK'] = dram['qkv_weightK'].load_mem(pt_weight, (param.hidden_size, range_weight))
            # sram_sp['qkv_weightV'] = dram['qkv_weightV'].load_mem(pt_weight, (param.hidden_size, range_weight))

            bit_out, sum_out, max_out = bitlinear.mem_control(sram_sp['qkv_token'], sram_sp['qkv_weightQ'])

            if pt_weight[1] == 0:
                out_token = bit_out.copy()
            else:
                out_token = [ t + b  for t, b in zip(out_token, bit_out)  ]

            sram_sp['qkv_buf1_Q'] = out_token
            pt_weight = (pt_weight[0], bitlinear.num_weight + pt_weight[1])
            pt_token  = (pt_token[0] + bitlinear.num_token, pt_token[1])
        
        # print(out_token)
        
        sram_sp['qkv_buf2_Q'] = []

        for j, token in enumerate(out_token):
            sram_sp['qkv_buf2_Q'].append(token)
            sum_out += token
            if j == 0 or token > max_out:
                max_out = token

        # sram_sp['q_token'] = [ token * 127 // max_out for token in sram_sp['qkv_buf2_Q'] ]
        # print(sram_sp['q_token'])
        sram_sp['q_token'] = []

        for token in sram_sp['qkv_buf2_Q']:
            sram_sp['q_token'].append([token * 127 // max_out])

        dram['quanQ'].store_mem((0, i), sram_sp['q_token'])

        out_seq.append(out_token.copy())
        out_max.append(max_out)
        out_sum.append(sum_out)

    out_seq = [ [ out_seq[j][i] for j in range(param.hidden_size) ] for i in range(param.hidden_size) ]

    print(f"########################################")
    print(f"###           TEST: OUTPUT           ###")
    print(f"########################################")

    dram['quanQ'].peak_mem()

    # SRAM Output
    output = {
        "out_sum": out_sum,
        "out_max": out_max,
        "out_seq": out_seq
    }
    peak_sram(output, "debug_QKV")

    for m, s in zip(out_max, out_sum):
        print(f"max {m:4d}, sum {s:4d}")


if __name__ == '__main__':
    print("This program is Executed")
    QKV_gen()
    # QKV_verify()

