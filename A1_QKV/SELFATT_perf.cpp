// SELFATT_perf.cpp
// Translated (structure-preserving) from SELFATT_perf.py
//
// Notes
// -----
// 1) This file preserves the *control / latency accounting* behavior of the Python script.
// 2) The numeric compute (QK/SMSM/SMD/QKV/QKVSM math) is not implemented here; only the
//    interfaces + latency hooks that SELFATT_perf.py depends on are modeled.
// 3) Replace the stub modules (QK/SMSM/SMD/QKV/QKVSM) with your real C++ implementations
//    when you are ready to validate correctness.
//
// Build (example)
// --------------
//   g++ -std=c++17 -O2 -o selfatt_perf SELFATT_perf.cpp
//
// Run
// ---
//   ./selfatt_perf

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

// ============================================================
// Configuration defaults (mirrors SELFATT_perf.py)
// ============================================================
static constexpr int SEQLEN_DEFAULT = 8192;
static constexpr int NUMHD_DEFAULT  = 32;
static constexpr const char* TEST_DEFAULT = "QKV"; // "QK", "SMSM", "SMD", "QKV"

// ============================================================
// Minimal stub modules
//   These exist ONLY so SELTATT can compile and run.
//   Replace with real implementations as needed.
// ============================================================

struct QK {
    int hiddenSize;
    int headNum;
    int seqLen;
    int headDim;
    int lat;

    explicit QK(int seqLen_,
                int hiddenSize_ = 8192,
                int headNum_    = 32,
                int lat_        = 4)
        : hiddenSize(hiddenSize_),
          headNum(headNum_),
          seqLen(seqLen_),
          headDim(hiddenSize_ / headNum_),
          lat(lat_) {}
};

struct SMSM {
    int seqLen;
    int lat;
    explicit SMSM(int seqLen_, int lat_ = 5) : seqLen(seqLen_), lat(lat_) {}
};

struct SMD {
    int seqLen;
    int lat;
    explicit SMD(int seqLen_, int lat_ = 6) : seqLen(seqLen_), lat(lat_) {}
};

struct QKV {
    int seqLen;
    int lat;

    // Control state used by SELTATT
    int  colCnt;   // maps to python self.instQKV.colCnt
    bool QKFlag;   // maps to python self.instQKV.QKFlag

    explicit QKV(int seqLen_, int lat_ = 7)
        : seqLen(seqLen_), lat(lat_), colCnt(0), QKFlag(true) {}

    void update_ctrl(int headDim) {
        // A minimal model:
        // - QKFlag is "true" when the unit is in "QK ingest" phase.
        // - Once QKFlag drops, we model streaming V columns by incrementing colCnt.
        //
        // This is NOT the real algorithm; it exists to let the latency/control
        // structure run end-to-end.
        if (QKFlag) {
            QKFlag = false;
            colCnt = 0;
        } else {
            if (colCnt < headDim - 1) colCnt++;
            else colCnt = headDim - 1; // clamp, SELTATT uses this terminal value
        }
    }
};

struct QKVSM {
    int seqLen;
    int lat;
    explicit QKVSM(int seqLen_, int lat_ = 3) : seqLen(seqLen_), lat(lat_) {}
};

// ============================================================
// SELTATT (ported from Python class SELTATT)
// ============================================================
class SELTATT {
public:
    // Settings
    int seqlen;
    std::string test;
    int numhd;

    // Control units
    int SMDCnt;
    int VColCnt;
    int QKCnt;
    int tkCnt;
    int qBufPter;
    int kBufPter;
    int vBufPter;
    int smSumPter;
    int smMaxPter;

    // Unroll variables
    int SMDQta;

    // Latency evaluation
    long long totalLat;
    long long latOvHdTmp1;
    long long latOvHdTmp2;
    long long latOvHdTmp3;
    long long latOvHd;

    // Instantiated submodules
    QK    instQK;
    SMSM  instSMSM;
    SMD   instSMD;
    QKV   instQKV;
    QKVSM instQKVSM;

    explicit SELTATT(int seqlen_ = SEQLEN_DEFAULT,
                     std::string test_ = TEST_DEFAULT,
                     int numhd_ = NUMHD_DEFAULT)
        : seqlen(seqlen_),
          test(std::move(test_)),
          numhd(numhd_),
          // control
          SMDCnt(0), VColCnt(0), QKCnt(0), tkCnt(0),
          qBufPter(0), kBufPter(0), vBufPter(0), smSumPter(0), smMaxPter(0),
          // unroll
          SMDQta(0),
          // latency
          totalLat(0), latOvHdTmp1(0), latOvHdTmp2(0), latOvHdTmp3(0), latOvHd(0),
          // modules
          instQK(seqlen_),
          instSMSM(seqlen_),
          instSMD(seqlen_),
          instQKV(seqlen_),
          instQKVSM(seqlen_) {

        // If user overrides numhd, reflect that in QK stub so headNum matches.
        instQK.headNum = numhd;
        instQK.headDim = instQK.hiddenSize / instQK.headNum;
    }

    void operations(bool verbose = true) {
        // Mirror:
        // for j in range(self.instQK.seqLen):
        //   self.totalLat += 1
        //   for i in range(self.instQK.headNum):
        //     self.one_operation(i, j)
        for (int j = 0; j < instQK.seqLen; ++j) {
            totalLat += 1; // load a row of q from DRAM (modeled)

            for (int i = 0; i < instQK.headNum; ++i) {
                one_operation(i, j, verbose);
            }
        }

        // Drain the rest of the SMD-QKV pipeline
        const bool stallFlag = (SMDQta > 1);
        while (SMDQta != 0) {
            // printf(" final round: SMDQta=%d\n", SMDQta);
            SMDQKV_call(/*mode=*/"rest", /*last=*/false, /*rowQ=*/31, verbose);
        }

        if (verbose) {
            std::cout << "\n---- Rest of the stages has terminate its work sucessfully ----\n";
        }

        // Final latency
        if (stallFlag) latOvHd = latOvHdTmp1 + latOvHdTmp2;
        else           latOvHd = latOvHdTmp1 + latOvHdTmp3;

        totalLat += latOvHd;

        if (verbose) {
            std::cout << "total latency count: " << totalLat << " cycles\n";
        }
    }

private:
    void one_operation(int headNumb, int rowQ, bool verbose = true) {
        // for l in range(self.instQK.headDim//32):
        for (int l = 0; l < instQK.headDim / 32; ++l) { // for every slice of q chunk
            totalLat += 1; // read q_deq, q and RoPE from SRAM (modeled)
            
            // Per slice, per (head,rowQ)
                totalLat += instQK.seqLen; // read k_deq, k and RoPE from SRAM (modeled) // all K rows
            
            if ((headNumb == instQK.headNum - 1) &&
                (rowQ     == instQK.seqLen - 1)  &&
                (l        == instQK.headDim / 32 - 1)
            ) {
                latOvHdTmp1 += instQK.lat; // checking
            }
            
            // l == last slice => SMSM stage (lat hook) and writeback hook
            if (test != "QK") {
                if (l   == instQK.headDim / 32 - 1 &&
                    headNumb == instQK.headNum - 1 &&
                    rowQ     == instQK.seqLen - 1
                ){
                    latOvHdTmp1 += (instSMSM.lat+1); // write SM_EXP to SRAM and the latency of last input for SMSM
                }
            }
            
            // for m in range(self.instQK.seqLen):
            // every row of partial vector from k
            const bool lastDur =
                (headNumb == instQK.headNum - 1) &&
                (rowQ    == instQK.seqLen - 1)   &&
                (l       == instQK.headDim / 32 - 1);

            if (SMDQta != 0) {
                SMDQKV_call(/*mode=*/"during", /*last=*/lastDur, /*rowQ=*/rowQ, verbose); // tbc unroll this line with m = seqlen times (concern, since the code is sharing with draining phase, the unroll revision should be conducted carefully)
            }

            // after processing all K rows for this slice
            // (python increments latOvHdTmp3 when lastDur at end of slice)
            // NOTE: python's lastDur depends on l==hiddenSize//32-1, so it is true only once.
            if (headNumb == instQK.headNum - 1 &&
                rowQ    == instQK.seqLen - 1 &&
                l       == instQK.hiddenSize / 32 - 1) {
                latOvHdTmp3 += 1; // write smExped into dram (modeled)
            }
        }

        // end of one softmax row -> enable SMD evaluation
        SMDQta += 1;
        (void)verbose; // silence unused warnings if verbose gated later
    }

    void SMDQKV_call(const std::string& mode, bool last, int rowQ, bool verbose = true) {
        // assert mode == "rest" or mode == "during"
        if (!(mode == "rest" || mode == "during")) {
            throw std::runtime_error("mode must be either \"rest\" or \"during\"");
        }

        const bool mdDuringTrig = last && (mode == "during");

        if (SMDCnt == 0) {
            // read smExped into sram, and V tile into sram (modeled as 1 cycle)
            if (mode == "rest")      latOvHdTmp2 += 1;
            else if (mdDuringTrig)   latOvHdTmp3 += 1;
        }

        // read SM_SUM / SM_MAX regs (modeled as 1 cycle)
        if (mode == "rest")      latOvHdTmp2 += 1;
        else if (mdDuringTrig)   latOvHdTmp3 += 1;

        // SMD stage (lat modeled)
        if (mode == "rest") {
            if (SMDQta == 1 && SMDCnt == (instQK.seqLen / 32) - 1 && QKCnt == instQK.seqLen - 1) {
                latOvHdTmp2 += instSMD.lat;
            } else {
                latOvHdTmp2 += 1;
            }
        } else if (mdDuringTrig) {
            latOvHdTmp3 += instSMD.lat;
        }

        if (test == "SMD" && verbose) {
            std::cout << "progress: NO." << QKCnt << "  " << (SMDCnt + 1)
                      << "/" << (instQK.seqLen / 32) << "\r" << std::flush;
        } else if (test == "QKV") {
            // if (!instQKV.QKFlag) {
                std::cout << "progress: QKVrow." << QKCnt
                          << ", head." << tkCnt
                          << ", seq slice." << SMDCnt
                          << ", input with V at NO." << VColCnt
                          << "/" << (instQK.hiddenSize / instQK.headNum - 1)
                          << "\r" << std::flush;
            // }

            // The python does:
            //   self.VColCnt = self.instQKV.colCnt
            //   self.instQKV.update_ctrl()
            if (mode == "rest"){
                VColCnt = instQKV.colCnt; //tbc when the mode is in during, it should have been execute for seqLen times (headDim+1 times to go around where the first and second time stay the same)
                instQKV.update_ctrl(instQK.hiddenSize / instQK.headNum);//tbc when the mode is in during, it should have been execute for seqLen times
            }
            
            if (mode == "rest") {
                if (SMDQta == 1 && SMDCnt == (instQK.seqLen / 32) - 1 && QKCnt == instQK.seqLen - 1) {
                    latOvHdTmp2 += instQKV.lat;
                    latOvHdTmp2 += instQKVSM.lat;
                    latOvHdTmp2 += 1; // write back t0 qkvSum and qkvMax SRAM
                } else {
                    latOvHdTmp2 += 1;
                }
            } else if (mdDuringTrig) {
                latOvHdTmp3 += instQKV.lat;
                latOvHdTmp3 += instQKVSM.lat;
                latOvHdTmp3 += 1; // write back t0 qkvSum and qkvMax SRAM
            }
        }

        // advance SMDCnt / tkCnt / QKCnt and consume quota
        //tbc when the mode is in during, it should have been execute for seqLen times
        if (mode == "rest"){
            const bool sliceDone = (VColCnt == (instQK.hiddenSize / instQK.headNum - 1)) || (test == "SMD");
            if (sliceDone) {
                if (SMDCnt == (instQK.seqLen / 32) - 1) {
                    SMDQta -= 1;
                    SMDCnt = 0;

                    if (tkCnt != instQK.headNum - 1) {
                        tkCnt += 1;
                    } else {
                        tkCnt = 0;

                        // if (mode == "during" && verbose) {
                        //     std::cout << "\n---- done row " << QKCnt << " ----\n";
                        // }

                        if (QKCnt != instQK.seqLen - 1) QKCnt += 1;
                        else                            QKCnt = 0;
                    }
                } else {
                    SMDCnt += 1;
                }
            }
        }
        else if (mode == "during") {
            if (SMDQta >= 32){ // trigger for seqLen times
                SMDQta -= 32;
            } else{
                tkCnt += SMDQta;
                if (tkCnt >= 32){
                    tkCnt -= 32;
                    QKCnt += 1;
                }

                SMDQta = 0;
            }
        }

    }
};

// ============================================================
// main (mirrors Python __main__ block)
// ============================================================
int main(int argc, char** argv) {
    int seqlen = SEQLEN_DEFAULT;
    int numhd  = NUMHD_DEFAULT;
    std::string test = TEST_DEFAULT;

    // Minimal CLI:
    //   ./selfatt_perf [seqlen] [test] [numhd]
    if (argc >= 2) seqlen = std::atoi(argv[1]);
    if (argc >= 3) test   = argv[2];
    if (argc >= 4) numhd  = std::atoi(argv[3]);

    SELTATT inst(seqlen, test, numhd);
    inst.operations(true);
    return 0;
}
