.PHONY: mat_mul mem_init mem_test mem_func LUT_init 

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
MEM_INIT = python3 -m A4_MEM.A1_INIT.${MEM} \
	--fill ${FILL} \
	--XLEN ${XLEN} \
	--YLEN ${YLEN} \
	--LNUM ${LNUM} \
	--name ${NAME}

MEM_TEST = python3 -m A4_MEM.A0_MEM.${MEM}
MEM_FUNC = python3 -m A4_MEM.A3_TEST.${MEM} --test ${TEST}

LUT_INIT = python3 -m A4_MEM.A1_INIT.${LUT} \
	--LNUM ${LNUM} \
	--name ${NAME}


######################################
# Make Command
######################################

#------------ Utilis -------------
mat_mul:
	cd ..  &&  ${MAT_MUL}

#-------------- MEM --------------
mem_init:
	cd ..  &&  ${MEM_INIT}
mem_test:
	cd ..  &&  ${MEM_TEST}
mem_func:
	cd ..  &&  ${MEM_FUNC}
LUT_init:
	cd ..  &&  ${LUT_INIT}