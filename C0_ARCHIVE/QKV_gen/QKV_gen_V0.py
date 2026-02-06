from A5_Utilis.A0_BITLINEAR.bitlinear import Bitlinear
from A5_Utilis.B0_CONFIG.global_param import param
from A4_MEM.A0_MEM.sram import sram_dp, sram_sp

from A4_MEM.A0_MEM.dram import dram
###################################################
###                MAIN CALLs                   ###
###################################################
Q_bitlinear = Bitlinear(param.hidden_size, param.hidden_size)
K_bitlinear = Bitlinear(param.hidden_size, param.hidden_size)
V_bitlinear = Bitlinear(param.hidden_size, param.hidden_size)

###################################################
###                TEST Program                 ###
###################################################
def QKV_gen():
    
    pt_weight = (0, 0);  range_weight = Q_bitlinear.num_weight;  len_weight = param.hidden_size // 8
    pt_token  = (0, 0);  range_token  = Q_bitlinear.num_token;   len_token  = param.hidden_size

    out_seq = []

    print(f"########################################")
    print(f"###          TEST: OUTPUT         ###")
    print(f"########################################")

    for i in range(param.seq_length):
        pt_weight = (0, 0);  range_weight = Q_bitlinear.num_weight;  len_weight = param.hidden_size // 8
        pt_token  = (0, i);  range_token  = Q_bitlinear.num_token;   len_token  = param.hidden_size

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

            # print(f"inside loop {pt_weight}, sram {pt_token} {range_token} {sram_sp['qkv_token'  ]}")
            # print(f"inside loop {pt_weight}, sram {pt_weight} {range_weight} {sram_sp['qkv_weightQ'  ]}")

            # print(f"{type(pt_weight[0])} / {type(Q_bitlinear.num_weight)}")
            bit_out, sum_out, max_out = Q_bitlinear.mem_control(sram_sp['qkv_token'], sram_sp['qkv_weightQ'])

            if pt_weight[1] == 0:
                out_token = bit_out.copy()
            else:
                out_token = [ t + b  for t, b in zip(out_token, bit_out)  ]
            
            pt_weight = (pt_weight[0], Q_bitlinear.num_weight + pt_weight[1])
            pt_token  = (pt_token[0] + Q_bitlinear.num_token, pt_token[1])

        print(out_token)

        out_seq.append(out_token.copy())

    # print(f"len {len(out_token)}")
    # print(out_token)
    # for token in out_seq:
    #     print(token)

if __name__ == '__main__':
    print("This program is Executed")
    QKV_gen()
    # QKV_verify()

