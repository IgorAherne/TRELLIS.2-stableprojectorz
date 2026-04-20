# File: trellis2/modules/sparse/conv/config.py
# trellis2/modules/sparse/conv/config.py
SPCONV_ALGO = 'auto'                               # 'auto', 'implicit_gemm', 'native'
FLEX_GEMM_ALGO = 'masked_implicit_gemm_splitk'     # 'explicit_gemm', 'implicit_gemm', 'implicit_gemm_splitk', 'masked_implicit_gemm', 'masked_implicit_gemm_splitk'
FLEX_GEMM_HASHMAP_RATIO = 1.5                      # Ratio of hashmap size to input size
