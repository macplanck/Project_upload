import numpy as np
from typing import Sequence
class QKVSM:

    def __init__(
        self,
        hiddenSize: int = 4096,
        headNum: int = 32,
        seqLen: int = 8192,
        lat: int = 2,
    ) -> None:
        # Floating-point exception behavior (as requested).
        np.seterr(over="raise", divide="raise", invalid="raise", under="ignore")

        # Model parameters
        self.hiddenSize = int(hiddenSize)
        self.headNum = int(headNum)
        self.seqLen = int(seqLen)
        self.lat = int(lat)

        # latency:
        # input buffering -> sum comparing -> max comparing
    
    def one_operation (self, inSum:np.float16, theSum: np.float16, theMax: np.float16, colMarker: int):
        
        finalSum, finalMax = self.eval(inSum=inSum, theSum=theSum, theMax=theMax, colMarker=colMarker)
        outValid = self.update_ctrl(colMarker=colMarker)

        return finalSum, finalMax, outValid
    
    def eval (self, inSum:Sequence[np.float16], theSum: np.float16, theMax: np.float16, colMarker: int):
        
        finalMax = inSum
        
        finalSum = inSum

        if (colMarker != 0):
            finalSum += theSum

            if (theMax > finalMax):
                finalMax = theMax

        return finalSum, finalMax

    def update_ctrl (self, colMarker):
        return colMarker == self.hiddenSize//self.headNum - 1
