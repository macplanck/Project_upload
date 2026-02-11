.PHONY: sram_sp_init sram_sp_test mem_init mem_test mem_func

######################################
# Set your desired file names
######################################
#-------------- SRAM --------------
path_mem = ../../04_MEM/00_MEM

######################################
# Default Setting
######################################


######################################
# Simulation Option and Command
######################################

# =====================================
# ============= SRAM ==================
# =====================================
SRAM_SP_INIT = python3 ${path_mem}/../01_INIT/${SRAM}.py \
	--fill ${FILL} \
	--ADDR ${ADDR} \
	--LNUM ${LNUM} \
	--name ${NAME}

SRAM_SP_TEST = python3 ${path_mem}/../01_SRAM/${SRAM}.py

MEM_INIT = python3 ${path_mem}/../01_INIT/${MEM}.py \
	--fill ${FILL} \
	--XLEN ${XLEN} \
	--YLEN ${YLEN} \
	--LNUM ${LNUM} \
	--name ${NAME}

MEM_TEST = python3 ${path_mem}/../00_MEM/${MEM}.py
MEM_FUNC = python3 ${path_mem}/../03_TEST/${MEM}.py

# =====================================
sram_sp_init:
	${SRAM_SP_INIT}

sram_sp_test:
	${SRAM_SP_TEST}

mem_init:
	${MEM_INIT}

mem_test:
	${MEM_TEST}

mem_func:
	${MEM_FUNC}

