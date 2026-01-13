## Global Parameters

#### File Paths
```
Root --- A5_Utilis --- B0_CONFIG --- config_mem.json
                                  |
                                  -- config_param.json
                                  |
                                  -- global_param.py
```

#### Global Parameters
- ***How to define***
  - Add your Parameters inside `config_param.json`: **eg.** `"hidden_size": 4096, `
  - Add your Parameters into `global_param.py` inside `class global_param`: **eg.** `self.hidden_size = self.config["hidden_size"]`
- ***How to get Parameters***
  - Import as following script **`from A5_Utilis.B0_CONFIG.global_param import param`**
  - Get your Global Parameters by **`param.your_parameter`**: **eg.** `param.hidden_size`