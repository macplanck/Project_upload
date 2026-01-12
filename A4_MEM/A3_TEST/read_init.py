from pathlib import Path

BASE = Path(__file__).resolve().parent
PATH_INIT = (BASE / "../01_INIT/dram_t.init")


def read_init():
    with open(PATH_INIT, 'r') as f:
        for i, line in enumerate(f):
            read_line = line.replace("\n", "")
            read_line = read_line.split(" ")
            while read_line.count('') > 0:
                read_line.remove('')
            read_line = [ int(element) for element in read_line ]
            # read_line = int(read_line)
            print(f"read line: {read_line}")

if __name__ == '__main__':
    read_init()
