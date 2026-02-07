import numpy as np
from A5_Utilis.A0_BITLINEAR.bitlinear import Bitlinear
from A5_Utilis.B0_CONFIG.global_param import param
from A5_Utilis.B1_CAL.integer import LOD
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

    q_seq_t = [];  q_sum_t = [];  q_max_t = [];  q_LOD_t = [];  q_exp_t = [];  q_mem_t = [];  q_scale_t = []
    k_seq_t = [];  k_sum_t = [];  k_max_t = [];  k_LOD_t = [];  k_exp_t = [];  k_mem_t = [];  k_scale_t = []
    v_seq_t = [];  v_sum_t = [];  v_max_t = [];  v_LOD_t = [];  v_exp_t = [];  v_mem_t = [];  v_scale_t = []

    # print(f"max, {sram_sp['qkv_max']};  sum, {sram_sp['qkv_max']}")
    # print(f"########################################")
    # print(f"###          TEST: OUTPUT         ###")
    # print(f"########################################")

    for i in range(param.seq_length):
        pt_weight = (0, 0);  range_weight = bitlinear.num_weight;  len_weight = param.hidden_size // 8
        pt_token  = (0, i);  range_token  = bitlinear.num_token;   len_token  = param.hidden_size

        q_token = [ 0 for val in sram_sp['qkv_token_Q'] ]
        k_token = [ 0 for val in sram_sp['qkv_token_K'] ]
        v_token = [ 0 for val in sram_sp['qkv_token_V'] ]

        # sum_scale, max_scale = sram_sp['qkv_sum'][i], sram_sp['qkv_max'][i]
        sum_scale, max_scale = bitlinear.scale(sram_sp['qkv_sum'][i], param.token_bit - sram_sp['qkv_max'][i] - 1)
        print(f"max: {max_scale:3d}, sum: {sum_scale}")

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

            # bit_out, sum_out, max_out = bitlinear.projection(sram_sp['qkv_token'], sram_sp['qkv_weightQ'])
            q_token, q_sum, q_max = bitlinear.projection(sram_sp['qkv_token'], sram_sp['qkv_weightQ'], q_token, sum_scale, max_scale)
            k_token, k_sum, k_max = bitlinear.projection(sram_sp['qkv_token'], sram_sp['qkv_weightK'], k_token, sum_scale, max_scale)
            v_token, v_sum, v_max = bitlinear.projection(sram_sp['qkv_token'], sram_sp['qkv_weightV'], v_token, sum_scale, max_scale)

            sram_sp['qkv_buf1_Q'] = q_token
            sram_sp['qkv_buf1_K'] = k_token
            sram_sp['qkv_buf1_V'] = v_token

            pt_weight = (pt_weight[0], bitlinear.num_weight + pt_weight[1])
            pt_token  = (pt_token[0] + bitlinear.num_token, pt_token[1])

        sram_sp['qkv_buf2_Q'] = []
        sram_sp['qkv_buf2_K'] = []
        sram_sp['qkv_buf2_V'] = []

        for qt, kt, vt in zip(sram_sp['qkv_buf1_Q'], sram_sp['qkv_buf1_K'], sram_sp['qkv_buf1_V']):
            sram_sp['qkv_buf2_Q'].append(qt)
            sram_sp['qkv_buf2_K'].append(kt)
            sram_sp['qkv_buf2_V'].append(vt)

        sram_sp['qkv_token_Q'] = []
        sram_sp['qkv_token_K'] = []
        sram_sp['qkv_token_V'] = []

        q_max_LOD = LOD(q_max);  q_max_exp = param.token_bit - q_max_LOD - 1;  q_max_mem = -q_max_exp - max_scale;  q_max_scale = 2 ** q_max_exp
        k_max_LOD = LOD(k_max);  k_max_exp = param.token_bit - k_max_LOD - 1;  k_max_mem = -k_max_exp - max_scale;  k_max_scale = 2 ** k_max_exp
        v_max_LOD = LOD(v_max);  v_max_exp = param.token_bit - v_max_LOD - 1;  v_max_mem = -v_max_exp - max_scale;  v_max_scale = 2 ** v_max_exp

        # print(f"scale {type(q_max_scale)} {q_max_scale},  ")
        
        for qt, kt, vt in zip(sram_sp['qkv_buf2_Q'], sram_sp['qkv_buf2_K'], sram_sp['qkv_buf2_V']):
            sram_sp['qkv_token_Q'].append([round(qt * q_max_scale)])
            sram_sp['qkv_token_K'].append([round(vt * k_max_scale)])
            sram_sp['qkv_token_V'].append([round(kt * v_max_scale)])

        dram['quanQ'].store_mem((0, i), sram_sp['qkv_token_Q'])
        dram['quanK'].store_mem((0, i), sram_sp['qkv_token_K'])
        dram['quanV'].store_mem((0, i), sram_sp['qkv_token_V'])

        sram_sp['qkv_max_Q'][i] = q_max_mem;  sram_sp['qkv_sum_Q'][i] = np.float16((q_sum ** 0.5) / (2 ** max_scale));  sram_sp['qkv_sum_Q'][i] = (q_sum ** 0.5) / (2 ** max_scale)
        sram_sp['qkv_max_K'][i] = k_max_mem;  sram_sp['qkv_sum_K'][i] = np.float16((k_sum ** 0.5) / (2 ** max_scale));  sram_sp['qkv_sum_K'][i] = (k_sum ** 0.5) / (2 ** max_scale)
        sram_sp['qkv_max_V'][i] = v_max_mem;  sram_sp['qkv_sum_V'][i] = np.float16((v_sum ** 0.5) / (2 ** max_scale));  sram_sp['qkv_sum_V'][i] = (v_sum ** 0.5) / (2 ** max_scale)

        ################################
        ###        TEST CODE         ###
        ################################
        q_seq_t.append(q_token.copy());  q_max_t.append(q_max);  q_sum_t.append(q_sum)
        k_seq_t.append(k_token.copy());  k_max_t.append(k_max);  k_sum_t.append(k_sum)
        v_seq_t.append(v_token.copy());  v_max_t.append(v_max);  v_sum_t.append(v_sum)

        q_LOD_t.append(q_max_LOD);  q_exp_t.append(q_max_exp);  q_scale_t.append(q_max_scale)
        k_LOD_t.append(k_max_LOD);  k_exp_t.append(k_max_exp);  k_scale_t.append(k_max_scale)
        v_LOD_t.append(v_max_LOD);  v_exp_t.append(v_max_exp);  v_scale_t.append(v_max_scale)

    q_seq_t = [ [ q_seq_t[j][i] for j in range(param.hidden_size) ] for i in range(param.hidden_size) ]
    k_seq_t = [ [ k_seq_t[j][i] for j in range(param.hidden_size) ] for i in range(param.hidden_size) ]
    v_seq_t = [ [ v_seq_t[j][i] for j in range(param.hidden_size) ] for i in range(param.hidden_size) ]

    print(f"########################################")
    print(f"###           TEST: OUTPUT           ###")
    print(f"########################################")

    dram['quanQ'].peak_mem()
    dram['quanK'].peak_mem()
    dram['quanV'].peak_mem()

    # SRAM Output
    # output = { "q_sum": q_sum_t,  "q_max": q_max_t,  "q_seq": q_seq_t };  peak_sram(output, "debug_QKV_Q")
    # output = { "k_sum": k_sum_t,  "k_max": k_max_t,  "k_seq": k_seq_t };  peak_sram(output, "debug_QKV_K")
    # output = { "v_sum": v_sum_t,  "v_max": v_max_t,  "v_seq": v_seq_t };  peak_sram(output, "debug_QKV_V")

    output = { "q_sum": q_sum_t,  "q_max": q_max_t,  "q_LOD": q_LOD_t,  "q_exp": q_exp_t,  "q_scale": q_scale_t,  "q_seq": q_seq_t };  peak_sram(output, "debug_QKV_Q")
    output = { "k_sum": k_sum_t,  "k_max": k_max_t,  "k_LOD": k_LOD_t,  "k_exp": k_exp_t,  "k_scale": k_scale_t,  "k_seq": k_seq_t };  peak_sram(output, "debug_QKV_K")
    output = { "v_sum": v_sum_t,  "v_max": v_max_t,  "v_LOD": v_LOD_t,  "v_exp": v_exp_t,  "v_scale": v_scale_t,  "v_seq": v_seq_t };  peak_sram(output, "debug_QKV_V")

    output = { "Q_max": sram_sp['qkv_max_Q'], "K_max": sram_sp['qkv_max_V'],  "V_max": sram_sp['qkv_max_V'] };  peak_sram(output, "debug_QKV_max")
    output = { "Q_sum": sram_sp['qkv_sum_Q'], "K_sum": sram_sp['qkv_sum_K'],  "V_max": sram_sp['qkv_sum_V'] };  peak_sram(output, "debug_QKV_sum")

if __name__ == '__main__':
    print("This program is Executed")
    QKV_gen()
    # QKV_verify()

