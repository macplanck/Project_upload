.PHONY: sram_sp_init sram_sp_test

######################################
# Set your desired file names
######################################
#-------------- SRAM --------------
path_mem = ../../04_MEM/01_SRAM

######################################
# Default Setting
######################################


######################################
# Simulation Option and Command
######################################

# =====================================
# ============= SRAM ==================
# =====================================
SRAM_SP_INIT = python3 ${path_mem}/../02_INIT/${SRAM}.py \
	--fill ${FILL} \
	--ADDR ${ADDR} \
	--LNUM ${LNUM} \
	--name ${NAME}

SRAM_SP_TEST = python3 ${path_mem}/../01_SRAM/${SRAM}.py

# =====================================
sram_sp_init:
	${SRAM_SP_INIT}

sram_sp_test:
	${SRAM_SP_TEST}
