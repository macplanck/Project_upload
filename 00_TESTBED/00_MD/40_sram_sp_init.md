
## Description
- Generate Initial Data into "SRAM_NAME_sp_init.txt"
---

#### initial type 
- Parameter Modify
  1. can only be change by modifying $in_LNUM

- Mode
fill zero         : 0
fill LUT          : 1
fill rand weight  : 2
fill rand token   : 3

---
#### Address Width   
- Parameter Modify
  1. Can only be change by modifying $in_ADDR

- NOTE
  1. Address width of Mem
  2. Can be neglect in **LUT mode**

---
#### < LUT NUM >   
- Parameter Modify
  1. Can only be change by modifying $in_LNUM

- NOTE
  1. Element num of LUT table 
---
#### SRAM Name  
> **SHOULD NOT BE BLANK !!!**
1. can be modified by command arguement. eg ./40_sram_sp_init SRAM_name
2. can be modified by changing $in_NAME
