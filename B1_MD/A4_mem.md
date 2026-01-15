## Add new Memory
1. **STEP 1**: Add **YOUR MEM** into `config.json`
```python!
  # Config PATH
  A5/B0_CONFIG --- config_dram.json
                |
                -- config_sram.json

  # MEM DATA
    "mem_name": {
        "mem_X": ROW_Length,
        "mem_Y": COL_Length,
        "type": "int" / "float",
        "sign": "signed" / "unsigned",
        "range": RANGE                     # [0, RANGE) (signed) / [-RANGE / 2, RANGE / 2) (unsigned) 
    }

```

2. **STEP 2**: Run Initialise Script inside `~/cmd/`
```
  ./41_mem_init
```


## Preparation Before Using MEM
1. **STEP 1**: Combine MEM FILES
```
  ./40_mem_combine
```
2. **STEP 2**: link `sram.py` from `A4_MEM/A1_INIT/` to `A4_MEM/A0_MEM/`
```
  cd ~/A4_MEM/A0_MEM/
  ln -s ../A1_INIT/sram.py
```

## Import MEMs into Desire FILE
- CHECK if FILE exist in `A4_MEM/A0_MEM/`
```
  dram.py
  sram.py       # link from A4_MEM/A1_INIT/
```

- **DRAM**
```
  from A4_MEM.A0_MEM.dram import dram
```
- **SRAM**
```
  from A4_MEM.A0_MEM.sram import sram
```

## Before Uploading to GitHub
- Remove `sram.py` and `dram.py` inside `A4_MEM/A1_INIT` (very important / DO NOT REMOVE THE WRONG FILE)
```
  ./49_mem_clean
```