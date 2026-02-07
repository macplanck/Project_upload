# ==================================================
# local.mk (Windows-only overrides; DO NOT COMMIT)
# ==================================================

# Choose the Python interpreter for Windows
PYTHON := /c/python_env/IClab/Scripts/python.exe

# Override the commands that currently use python3
QKV_GEN      = $(PYTHON) -m A1_QKV.${FILE} | tee ${LOG}
MEM_INIT     = $(PYTHON) -m A4_MEM.A1_INIT.${MEM} \
	--XLEN  ${XLEN}  \
	--YLEN  ${YLEN}  \
	--type  ${TYPE}  \
	--sign  ${SIGN}  \
	--name  ${NAME}  \
	--range ${RANGE}

MEM_TEST     = $(PYTHON) -m A4_MEM.A0_MEM.${MEM}
MEM_FUNC     = $(PYTHON) -m A4_MEM.A3_TEST.${MEM} --test ${TEST}

LUT_INIT     = $(PYTHON) -m A4_MEM.A1_INIT.${LUT} \
	--LNUM ${LNUM} \
	--name ${NAME}

SELFATT_TEST = $(PYTHON) -m A2_SELFATT.${FILE} | tee ${LOG}
MLP_TEST     = $(PYTHON) -m A3_MLP.${FILE} | tee ${LOG}

MAT_MUL      = $(PYTHON) -m A5_Utilis.A0_BITLINEAR.${FILE}
MAT_VER      = $(PYTHON) -m A5_Utilis.A0_BITLINEAR.${FILE} \
	--f ${NAME}   \
	--t ${TOKEN}  \
	--w ${WEIGHT} | tee ${LOG}

TEST         = $(PYTHON) -m B2_TEST.${FILE}
