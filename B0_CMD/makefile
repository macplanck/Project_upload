.PHONY: mat_mul mem_combine mem_clean mem_init mem_test mem_zip mem_func LUT_init 

######################################
# Set your desired file names
######################################
#-------------- MEM --------------
path_mem = A4_MEM/A0_MEM

######################################
# Default Setting
######################################


######################################
# Simulation Option and Command
######################################

#---------- Bitlinear ------------
MAT_MUL = python3 -m A5_Utilis.A0_BITLINEAR.${FILE}

#-------------- MEM --------------
<<<<<<< HEAD
DRAM_COMB = cat A4_MEM/A1_INIT/dram.def A4_MEM/A1_INIT/dram_*.init > A4_MEM/A1_INIT/dram.py
SRAM_COMB = cat A4_MEM/A1_INIT/sram.def A4_MEM/A1_INIT/sram_*.init > A4_MEM/A1_INIT/sram.py
=======
# DRAM_COMB = type A4_MEM/A1_INIT/dram.def A4_MEM/A1_INIT/dram_*.init > A4_MEM/A1_INIT/dram.py
# SRAM_COMB = type A4_MEM/A1_INIT/sram.def A4_MEM/A1_INIT/sram_*.init > A4_MEM/A1_INIT/sram.py

DRAM_COMB = powershell -Command "Get-Content A4_MEM/A1_INIT/dram.def, A4_MEM/A1_INIT/dram_*.init | Set-Content A4_MEM/A1_INIT/dram.py"
SRAM_COMB = powershell -Command "Get-Content A4_MEM/A1_INIT/sram.def, A4_MEM/A1_INIT/sram_*.init | Set-Content A4_MEM/A1_INIT/sram.py"
>>>>>>> orgin/main

MEM_CLEAN = rm -rf A4_MEM/A1_INIT/dram.py A4_MEM/A1_INIT/sram.py

MEM_INIT = python3 -m A4_MEM.A1_INIT.${MEM} \
	--XLEN  ${XLEN}  \
	--YLEN  ${YLEN}  \
	--type  ${TYPE}  \
	--sign  ${SIGN}  \
	--name  ${NAME}  \
<<<<<<< HEAD
	--range ${RANGE}
=======
	--range ${RANGE} \
	--bias  ${BIAS}
>>>>>>> orgin/main

MEM_TEST = python3 -m A4_MEM.A0_MEM.${MEM}
MEM_FUNC = python3 -m A4_MEM.A3_TEST.${MEM} --test ${TEST}

LUT_INIT = python3 -m A4_MEM.A1_INIT.${LUT} \
	--LNUM ${LNUM} \
	--name ${NAME}


######################################
# Make Command
######################################

#------------ Utilis -------------
<<<<<<< HEAD
mat_mul:
	cd ..  &&  ${MAT_MUL}
=======
# mat_mul:
# 	cd ..  &&  ${MAT_MUL}
>>>>>>> orgin/main

#-------------- MEM --------------
mem_combine:
	cd ..  &&  ${DRAM_COMB}
	cd ..  &&  ${SRAM_COMB}
mem_clean:
	cd ..  &&  ${MEM_CLEAN}
mem_init:
	cd ..  &&  ${MEM_INIT}
mem_test:
	cd ..  &&  ${MEM_TEST}
mem_func:
	cd ..  &&  ${MEM_FUNC}
LUT_init:
	cd ..  &&  ${LUT_INIT}