import numpy as np

BIAS_F16 = 15

<<<<<<< HEAD
###################################################
###                TEST Program                 ###
###################################################
def f16_exp(x):
    """
    RETURNs
      sign_bit: 0/1
      exp_field: 0 ~ 31 (exponent)
      frac_field: 0 ~ 1023 (mantissa)
      exp_unbiased: exponent without bias; inf/nan return None
      mantissa: normal = 1 + frac/2^10, subnormal = frac/2^10
=======
def f16_parts(x):
    """
    回傳：
      sign_bit: 0/1
      exp_field: 0..31 (原始 exponent 欄位)
      frac_field: 0..1023 (原始 mantissa/fraction 欄位)
      exp_unbiased: 指數(去 bias)；對 inf/nan 回傳 None
      mantissa: 尾數；normal = 1 + frac/2^10，subnormal = frac/2^10
>>>>>>> orgin/main
      kind: 'zero'/'subnormal'/'normal'/'inf'/'nan'
    """
    a = np.asarray(x, dtype=np.float16)
    bits = a.view(np.uint16)

    sign_bit   = (bits >> 15) & 0x1
    exp_field  = (bits >> 10) & 0x1F
    frac_field = bits & 0x3FF

<<<<<<< HEAD
    # unbiased exponent / mantissa
=======
    # 分類與計算 unbiased exponent / mantissa
>>>>>>> orgin/main
    exp_unbiased = np.full(exp_field.shape, None, dtype=object)
    mantissa     = np.zeros(exp_field.shape, dtype=np.float64)
    kind         = np.empty(exp_field.shape, dtype=object)

    is_exp0  = (exp_field == 0)
    is_exp31 = (exp_field == 31)
    is_frac0 = (frac_field == 0)

    # zero
    m = is_exp0 & is_frac0
    kind[m] = "zero"
<<<<<<< HEAD
    exp_unbiased[m] = 1 - BIAS_F16  
=======
    exp_unbiased[m] = 1 - BIAS_F16  # 慣例上 subnormal/zero 指數用 1-bias
>>>>>>> orgin/main
    mantissa[m] = 0.0

    # subnormal
    m = is_exp0 & (~is_frac0)
    kind[m] = "subnormal"
    exp_unbiased[m] = 1 - BIAS_F16
    mantissa[m] = frac_field[m] / 2**10

    # inf
    m = is_exp31 & is_frac0
    kind[m] = "inf"
    exp_unbiased[m] = None
    mantissa[m] = np.inf

    # nan
    m = is_exp31 & (~is_frac0)
    kind[m] = "nan"
    exp_unbiased[m] = None
    mantissa[m] = np.nan

    # normal
    m = (~is_exp0) & (~is_exp31)
    kind[m] = "normal"
    exp_unbiased[m] = (exp_field[m].astype(np.int32) - BIAS_F16)
    mantissa[m] = 1.0 + frac_field[m] / 2**10

<<<<<<< HEAD
    if a.shape == ():
        return (int(sign_bit), int(exp_field), int(frac_field), exp_unbiased.item(), float(mantissa), kind.item())
    return sign_bit, exp_field, frac_field, exp_unbiased, mantissa, kind


def f16_parts(x):
    """
    RETURNs
      sign_bit: 0/1
      exp_field: 0 ~ 31 (exponent)
      frac_field: 0 ~ 1023 (mantissa)
      exp_unbiased: exponent without bias; inf/nan return None
      mantissa: normal = 1 + frac/2^10, subnormal = frac/2^10
      kind: 'zero'/'subnormal'/'normal'/'inf'/'nan'
    """
    a = np.asarray(x, dtype=np.float16)
    bits = a.view(np.uint16)

    sign_bit   = (bits >> 15) & 0x1
    exp_field  = (bits >> 10) & 0x1F
    frac_field = bits & 0x3FF

    # unbiased exponent / mantissa
    exp_unbiased = np.full(exp_field.shape, None, dtype=object)
    kind         = np.empty(exp_field.shape, dtype=object)

    is_exp0  = (exp_field == 0)
    is_exp31 = (exp_field == 31)
    is_frac0 = (frac_field == 0)

    # zero
    m = is_exp0 & is_frac0
    kind[m] = "zero"
    exp_unbiased[m] = 1 - BIAS_F16  

    # subnormal
    m = is_exp0 & (~is_frac0)
    kind[m] = "subnormal"
    exp_unbiased[m] = 1 - BIAS_F16

    # inf
    m = is_exp31 & is_frac0
    kind[m] = "inf"
    exp_unbiased[m] = None

    # nan
    m = is_exp31 & (~is_frac0)
    kind[m] = "nan"
    exp_unbiased[m] = None

    # normal
    m = (~is_exp0) & (~is_exp31)
    kind[m] = "normal"
    exp_unbiased[m] = (exp_field[m].astype(np.int32) - BIAS_F16)

    if a.shape == ():
        return (exp_unbiased.item(), kind.item())
    return exp_unbiased, kind




###################################################
###                TEST Program                 ###
###################################################
if __name__ == '__main__':
    print("test")
=======
    # 如果輸入是 scalar，輸出也變 scalar
    if a.shape == ():
        return (int(sign_bit), int(exp_field), int(frac_field),
                exp_unbiased.item(), float(mantissa), kind.item())
    
    return sign_bit, exp_field, frac_field, exp_unbiased, mantissa, kind
>>>>>>> orgin/main
