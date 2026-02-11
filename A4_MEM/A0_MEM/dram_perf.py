from pathlib import Path
from A5_Utilis.B0_CONFIG.read_config import read_config
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
###                DRAM CLASS                   ###
###################################################
class DRAM:
    def __init__(self, config_in, name):
        self.dram_X = config_in["dram_X"]
        self.dram_Y = config_in["dram_Y"]
        self.file_init  = ( PATH_INIT / f"dram_{name}.csv" )
        self.file_out   = ( PATH_OUT  / f"dram_{name}.csv" )
        # self.file_deb   = ( PATH_DEB  / f"dram_{name}.csv" )
        # self.memory = dram_t[f'{name}']
        self.name = name

    # def peak_mem(self, start=None, end=None, file=None, info='PEAK'):

    #     if start == None:
    #         start = (0, 0)
    #     if end == None:
    #         end = (self.dram_X, self.dram_Y)
    #     if file == None:
    #         file = self.file_out

    #     if end[0] > self.dram_X:
    #         raise ValueError(f"MemError in op PEAK {self.name}: X Address Exceed Valid Range")
    #     elif end[1] > self.dram_Y:
    #         raise ValueError(f"MemError in op PEAK {self.name}: Y Address Exceed Valid Range")

    #     with open(file, 'w') as f:
    #         f.write(f"{info}")
    #         for i in range(start[1], end[1]):
    #             f.write(f",{i}")
    #         f.write("\n")

    #         for i in range(start[0], end[0]):
    #             f.write(f"{i}")
    #             for j in range(start[1], end[1]):
    #                 f.write(f",{self.memory[i][j]}")
    #             f.write("\n")
    #         print(f"Output Data exported to \'{file}\' ...")
    
    # def store_mem(self, start, sram_in):
        
    #     if isinstance(sram_in[0], list):    # if SRAM is 2D array
    #         sram = sram_in.copy()
    #     else:                               # if SRAM is 1D array
    #         sram = [ sram_in.copy() ]
        
    #     Range = (len(sram), len(sram[0]))

    #     # print(f"Input SRAM: {sram}")

    #     if start[0] + Range[0] > self.dram_X:
    #         raise ValueError(f"MemError in op STORE {self.name}: X Address Exceed Valid Range")
    #     elif start[1] + Range[1] > self.dram_Y:
    #         raise ValueError(f"MemError in op STORE {self.name}: Y Address Exceed Valid Range")

    #     for i in range(Range[0]):
    #         for j in range(Range[1]):
    #             self.memory[i + start[0]][j + start[1]] = sram[i][j]

    # def load_mem(self, start, Range):

    #     sram_out = []
    #     temp_out = []

    #     if isinstance(Range, tuple):    # if SRAM is 2D array

    #         end = (start[0] + Range[0], start[1] + Range[1])

    #         if end[0] > self.dram_X:
    #             raise ValueError(f"MemError in op LOAD {self.name}: X Address Exceed Valid Range")
    #         elif end[1] > self.dram_Y:
    #             raise ValueError(f"MemError in op LOAD {self.name}: Y Address Exceed Valid Range")
            
    #         for i in range(start[0], end[0]):
    #             for j in range(start[1], end[1]):
    #                 temp_out.append(self.memory[i][j])
    #             sram_out.append(temp_out.copy())
    #             temp_out.clear()
    #     else:                           # if SRAM is 1D array

    #         end = start[1] + Range
    #         if end > self.dram_Y:
    #             raise ValueError(f"MemError in op PEAK {self.name}: Y Address Exceed Valid Range")
            
    #         for j in range(start[1], end):
    #             sram_out.append(self.memory[start[0]][j])

    #     return sram_out.copy()

###################################################
###             GLOBAL Functions                ###
###################################################
def dram_init():

    global dram; dram = {}
    mem_config = read_config(PATH_CONF)

    for item in mem_config:
        dram[item] = DRAM(mem_config[item], item)
    
###################################################
###                MAIN CALLs                   ###
###################################################
dram_init()

###################################################
###                TEST Program                 ###
###################################################
if __name__ == '__main__':
    for item in dram:
        dram[item].peak_mem()


