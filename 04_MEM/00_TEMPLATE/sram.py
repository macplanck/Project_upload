from pathlib import Path

BASE = Path(__file__).resolve().parent
ADDR_WIDTH = 1024
sram_sp_name = 'sram'

class SRAM_SP:

    def __init__(self, name):

        global ADDR_WIDTH

        self.Memory = []
        self.PATH_INIT = (BASE / f"../02_INIT/{name}_sp_init.txt")
        self.PATH_OUT  = (BASE / f"../03_OUT/{name}_sp_out.txt")
        self.__name = name
        
        with open(self.PATH_INIT, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    ADDR_WIDTH = int(line)
                    print(f"ADDR: {ADDR_WIDTH}")
                else:
                    self.Memory.append(int(line))
        
        self.ADDR_WIDTH = ADDR_WIDTH

    def mem_read(self, addr_in):
        if addr_in >= self.ADDR_WIDTH:
            raise ValueError(f"MemError in {self.__name}: Input Address over Address width in Mem Read")
        else:
            return self.Memory[addr_in]
        
    def mem_write(self, addr_in, data_in):
        if addr_in >= self.ADDR_WIDTH:
            raise ValueError(f"MemError in {self.__name}: Input Address over Address width in Mem Write")
        else:
            self.Memory[addr_in] = data_in

    def mem_load(self, data_in, start=0):

        if len(data_in) + start >= self.ADDR_WIDTH:
            raise ValueError(f"MemError in {self.__name}: Load data over Address width")
        for i, data in enumerate(data_in):
            self.Memory[i+start] = data

    def peak_value(self, start=0, end=ADDR_WIDTH):

        with open(self.PATH_OUT, 'w') as f:
            f.write("**************************\n")
            f.write("  ADDR  ***   Value    ***\n")
            f.write("**************************\n")

            for i in range(start, end):
                f.write(f"{i:5d}    |    {self.Memory[i]:5d}      |\n")
            

if __name__ == '__main__':
    test_sram_sp = SRAM_SP(name=sram_sp_name)
    test_sram_sp.peak_value(end=ADDR_WIDTH)
    print(f"output file generated to \'{sram_sp_name}\' ...")


