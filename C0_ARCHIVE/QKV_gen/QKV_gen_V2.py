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

    q_seq = [];  q_sum = [];  q_max = []
    k_seq = [];  k_sum = [];  k_max = []
    v_seq = [];  v_sum = [];  v_max = []

    # print(f"########################################")
    # print(f"###          TEST: OUTPUT         ###")
    # print(f"########################################")

    for i in range(param.seq_length):
        pt_weight = (0, 0);  range_weight = bitlinear.num_weight;  len_weight = param.hidden_size // 8
        pt_token  = (0, i);  range_token  = bitlinear.num_token;   len_token  = param.hidden_size

        sum_out = 0
        max_out = 0

        while pt_weight[1] < len_weight:
        
            ################ [LOAD SRAM] ################
            if pt_token[0]  + range_token  > len_token:
                range_token = param.hidden_size - pt_token[0]
            if pt_weight[1] + range_weight > len_weight:
                range_weight = param.hidden_size - pt_weight[0]

            sram_sp['qkv_token'  ] = dram['qkv_token'  ].load_mem(pt_token , (range_token, 1))
            sram_sp['qkv_weightQ'] = dram['qkv_weightQ'].load_mem(pt_weight, (param.hidden_size, range_weight))
            sram_sp['qkv_weightK'] = dram['qkv_weightK'].load_mem(pt_weight, (param.hidden_size, range_weight))
            sram_sp['qkv_weightV'] = dram['qkv_weightV'].load_mem(pt_weight, (param.hidden_size, range_weight))

            # bit_out, sum_out, max_out = bitlinear.mem_control(sram_sp['qkv_token'], sram_sp['qkv_weightQ'])
            q_bit = bitlinear.mem_control(sram_sp['qkv_token'], sram_sp['qkv_weightQ'])
            k_bit = bitlinear.mem_control(sram_sp['qkv_token'], sram_sp['qkv_weightK'])
            v_bit = bitlinear.mem_control(sram_sp['qkv_token'], sram_sp['qkv_weightV'])

            if pt_weight[1] == 0:
                q_token = q_bit.copy()
                k_token = k_bit.copy()
                v_token = v_bit.copy()
            else:
                q_token = [ t + b  for t, b in zip(q_token, q_bit)  ]
                k_token = [ t + b  for t, b in zip(k_token, k_bit)  ]
                v_token = [ t + b  for t, b in zip(v_token, v_bit)  ]

            sram_sp['qkv_buf1_Q'] = q_token
            sram_sp['qkv_buf1_K'] = k_token
            sram_sp['qkv_buf1_V'] = v_token

            pt_weight = (pt_weight[0], bitlinear.num_weight + pt_weight[1])
            pt_token  = (pt_token[0] + bitlinear.num_token, pt_token[1])
        
        sram_sp['qkv_buf2_Q'] = [];  q_sum_t = 0;  q_max_t = q_token[0]
        sram_sp['qkv_buf2_K'] = [];  k_sum_t = 0;  k_max_t = k_token[0]
        sram_sp['qkv_buf2_V'] = [];  v_sum_t = 0;  v_max_t = v_token[0]

        for qt, kt, vt in zip(q_token, k_token, v_token):
            sram_sp['qkv_buf2_Q'].append(qt);  q_sum_t += qt
            sram_sp['qkv_buf2_K'].append(kt);  k_sum_t += kt
            sram_sp['qkv_buf2_V'].append(vt);  v_sum_t += vt
            
            if qt > q_max_t:
                q_max_t = qt
            if kt > k_max_t:
                k_max_t = kt
            if vt > v_max_t:
                v_max_t = vt

        # sram_sp['q_token'] = [ token * 127 // max_out for token in sram_sp['qkv_buf2_Q'] ]
        # print(sram_sp['q_token'])
        sram_sp['qkv_token_Q'] = []
        sram_sp['qkv_token_K'] = []
        sram_sp['qkv_token_V'] = []
        
        for qt, kt, vt in zip(sram_sp['qkv_buf2_Q'], sram_sp['qkv_buf2_K'], sram_sp['qkv_buf2_V']):
            sram_sp['qkv_token_Q'].append([qt * 127 // q_max_t])
            sram_sp['qkv_token_K'].append([vt * 127 // k_max_t])
            sram_sp['qkv_token_V'].append([kt * 127 // v_max_t])

        dram['quanQ'].store_mem((0, i), sram_sp['qkv_token_Q'])
        dram['quanK'].store_mem((0, i), sram_sp['qkv_token_K'])
        dram['quanV'].store_mem((0, i), sram_sp['qkv_token_V'])

        ################################
        ###         TEST CODE        ###
        ################################
        q_seq.append(q_token.copy());  q_max.append(q_max_t);  q_sum.append(q_sum_t)
        k_seq.append(k_token.copy());  k_max.append(k_max_t);  k_sum.append(k_sum_t)
        v_seq.append(v_token.copy());  v_max.append(v_max_t);  v_sum.append(v_sum_t)
        # out_sum.append(sum_out)

    q_seq = [ [ q_seq[j][i] for j in range(param.hidden_size) ] for i in range(param.hidden_size) ]
    k_seq = [ [ k_seq[j][i] for j in range(param.hidden_size) ] for i in range(param.hidden_size) ]
    v_seq = [ [ v_seq[j][i] for j in range(param.hidden_size) ] for i in range(param.hidden_size) ]

    print(f"########################################")
    print(f"###           TEST: OUTPUT           ###")
    print(f"########################################")

    dram['quanQ'].peak_mem()
    dram['quanK'].peak_mem()
    dram['quanV'].peak_mem()

    # SRAM Output
    output = { "q_sum": q_sum,  "q_max": q_max,  "q_seq": q_seq };  peak_sram(output, "debug_QKV_Q")
    output = { "k_sum": k_sum,  "k_max": k_max,  "k_seq": k_seq };  peak_sram(output, "debug_QKV_K")
    output = { "v_sum": v_sum,  "v_max": v_max,  "v_seq": v_seq };  peak_sram(output, "debug_QKV_V")


if __name__ == '__main__':
    print("This program is Executed")
    QKV_gen()
    # QKV_verify()

