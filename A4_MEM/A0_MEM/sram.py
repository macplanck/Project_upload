from pathlib import Path
from A4_MEM.A1_INIT.sram import sram_dp_t, sram_sp_t
from A5_Utilis.B2_PERF.perf import PERF
###################################################
###            GLOBAL PARAMETERS                ###
###################################################
BASE = Path(__file__).resolve().parent
PATH_CONF = (BASE / "../../A5_Utilis/B0_CONFIG/config_dram.json")
PATH_INIT = (BASE / "../A1_INIT")
PATH_OUT  = (BASE / "../A2_OUT")

if PERF:
    PATH_CONF = (BASE / "../../A5_Utilis/B2_PERF/config_dram.json")

###################################################
###                MAIN CALLs                   ###
###################################################
sram_dp = sram_dp_t
sram_sp = sram_sp_t


# def peak_sram(sram_in, file='sram_out', info='PEAK'):

#     if not isinstance(sram_in[0], list):
#         memory = [ sram_in.copy() ]
#     else:
#         memory = sram_in.copy()

#     file = f"{PATH_OUT}/{file}.csv"
#     sram_X = len(sram_in)
#     sram_Y = len(sram_in[0])
    
#     with open(file, 'w') as f:
#         f.write(f"{info}")
#         for i in range(sram_Y):
#             f.write(f",{i}")
#         f.write("\n")

#         for i in range(sram_X):
#             f.write(f"{i}")
#             for j in range(sram_Y):
#                 f.write(f",{memory[i][j]}")
#             f.write("\n")
#         print(f"Output Data exported to \'{file}\' ...")


def peak_sram(sram_in, file='sram_out'):

    file = f"{PATH_OUT}/{file}.csv"

    # for item in sram_in:
    #     print(item)

    with open(file, 'w') as f:

        for item in sram_in:
            # print(f"sram_type {item} {type(sram_in[item])}")
            if not isinstance(sram_in[item][0], list):
                memory = [ sram_in[item].copy() ]
            else:
                memory = sram_in[item].copy()

            sram_X = len(memory)
            sram_Y = len(memory[0])
        
            f.write(f"{item}")
            for i in range(sram_Y):
                f.write(f",{i}")
            f.write("\n")

            for i in range(sram_X):
                f.write(f"{i}")
                for j in range(sram_Y):
                    f.write(f",{memory[i][j]}")
                f.write("\n")
            f.write(",\n")
        print(f"Output Data exported to \'{file}\' ...")
