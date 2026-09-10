import numpy as np

# --------------------------------
# GEMM INPUTS
# --------------------------------

def generate_gemm_inputs():

    x = np.random.randn(4, 4).astype(np.float32)
    y = np.random.randn(4, 4).astype(np.float32)

    np.save("../models/inputs/gemm_x.npy", x)
    np.save("../models/inputs/gemm_y.npy", y)

    print("GEMM inputs generated")


# --------------------------------
# MLP INPUT
# --------------------------------

def generate_mlp_input():

    x = np.random.randn(1, 128).astype(np.float32)

    np.save("../models/inputs/mlp_input.npy", x)

    print("MLP input generated")


# --------------------------------
# MAIN
# --------------------------------

if __name__ == "__main__":

    generate_gemm_inputs()
    generate_mlp_input()
