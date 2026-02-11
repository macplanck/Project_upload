from A5_Utilis.B0_CONFIG.global_param import param


def cycle_cal(input_size, output_size):

    available_cycles = param.seq_length * (param.head_dim + 1)
    require_accesses = input_size * output_size * param.weight_bit // param.LUT_NUM + input_size * param.token_bit

    print(f"available cycles: {available_cycles}")
    print(f"require accesses: {require_accesses}")

    access_bits = require_accesses / available_cycles

    print(f"access bit: {access_bits}")

    vec_weight_len = round(access_bits / param.weight_bit + 0.5)
    vec_token_len = vec_weight_len * param.LUT_NUM

    print(f"vec_weight: {vec_weight_len}")
    print(f"vec_token: {vec_token_len}")

    size_weight = input_size
    size_token = input_size

    start_point = vec_token_len

    while vec_token_len

if __name__ == '__main__':
    cycle_cal(param.hidden_size, param.intermediate_size)