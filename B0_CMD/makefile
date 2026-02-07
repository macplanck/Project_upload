.PHONY: qkv_gen mlp_test mat_mul mat_verify mem_combine mem_init mem_test mem_func test clean

###################################################
###             GLOBAL Parameters               ###
###################################################
path_mem = A4_MEM/A0_MEM

###################################################
###            Simulation Command               ###
###################################################
#-------------- QKV --------------
QKV_GEN = python3 -m A1_QKV.${FILE} | tee ${LOG}

#-------------- MEM --------------
DRAM_COMB = cat A4_MEM/A1_INIT/dram.def A4_MEM/A1_INIT/dram_*.init > A4_MEM/A1_INIT/dram.py
SRAM_COMB = cat A4_MEM/A1_INIT/sram.def A4_MEM/A1_INIT/sram_*.init > A4_MEM/A1_INIT/sram.py

MEM_INIT = python3 -m A4_MEM.A1_INIT.${MEM} \
	--XLEN  ${XLEN}  \
	--YLEN  ${YLEN}  \
	--type  ${TYPE}  \
	--sign  ${SIGN}  \
	--name  ${NAME}  \
	--range ${RANGE}

MEM_TEST = python3 -m A4_MEM.A0_MEM.${MEM}
MEM_FUNC = python3 -m A4_MEM.A3_TEST.${MEM} --test ${TEST}

LUT_INIT = python3 -m A4_MEM.A1_INIT.${LUT} \
	--LNUM ${LNUM} \
	--name ${NAME}

#-------------- MLP --------------
MLP_TEST = python3 -m A3_MLP.${FILE} | tee ${LOG}

#------------ Utilis -------------
MAT_MUL = python3 -m A5_Utilis.A0_BITLINEAR.${FILE}
MAT_VER = python3 -m A5_Utilis.A0_BITLINEAR.${FILE} \
	--f ${NAME}   \
	--t ${TOKEN}   \
	--w ${WEIGHT}  | tee ${LOG}

#-------------- Else --------------
TEST = python3 -m B2_TEST.${FILE}

CLEAN = rm -rf \
	${DIR}/A1_INIT/*.${FILE} \
	${DIR}/A2_OUT/*.${FILE} \
	${DIR}/A1_INIT/dram.py${FILE} \
	${DIR}/A1_INIT/sram.py${FILE} \
	${DIR}/*.${FILE}

###################################################
###                MAKE Command                 ###
###################################################
#-------------- QKV --------------
qkv_gen:
	cd ..  &&  ${QKV_GEN}

#-------------- MLP --------------
mlp_test:
	cd ..  &&  ${MLP_TEST}

#------------ Utilis -------------
mat_mul:
	cd ..  &&  ${MAT_MUL}
mat_verify:
	cd ..  &&  ${MAT_VER}
#-------------- MEM --------------
mem_combine:
	cd ..  &&  ${DRAM_COMB}
	cd ..  &&  ${SRAM_COMB}

mem_init:
	cd ..  &&  ${MEM_INIT}
mem_test:
	cd ..  &&  ${MEM_TEST}
mem_func:
	cd ..  &&  ${MEM_FUNC}

#-------------- Else --------------
test:
	cd ..  &&  ${TEST}
clean:
	cd ..  &&  ${CLEAN}