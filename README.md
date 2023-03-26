# hf_model_manip

## Need this patch
```
--- /data/transformers/src/transformers/models/bloom/configuration_bloom.py     2023-03-18 08:12:05.422710148 +0800
+++ configuration_bloom.py      2023-03-26 16:48:16.135256708 +0800
@@ -116,6 +118,7 @@
         hidden_size=64,
         n_layer=2,
         n_head=8,
+        up_scale_factor=4, # FIXME
         layer_norm_epsilon=1e-5,
         initializer_range=0.02,
         use_cache=True,
@@ -134,6 +137,7 @@
         self.hidden_size = hidden_size if n_embed is None else n_embed
         self.n_layer = n_layer
         self.n_head = n_head
+        self.up_scale_factor = up_scale_factor # FIXME
         self.layer_norm_epsilon = layer_norm_epsilon
         self.initializer_range = initializer_range
         self.use_cache = use_cache
```

```
--- /data/transformers/src/transformers/models/bloom/modeling_bloom.py  2023-03-18 08:12:05.422710148 +0800                                                                                                           
+++ modeling_bloom.py   2023-03-26 16:48:25.031929823 +0800                                                
@@ -375,9 +376,9 @@                                                                                        
                                                  
         self.pretraining_tp = config.pretraining_tp
         self.slow_but_exact = config.slow_but_exact                                                       
-        self.dense_h_to_4h = nn.Linear(hidden_size, 4 * hidden_size)                                      
+        self.dense_h_to_4h = nn.Linear(hidden_size, int(config.up_scale_factor * hidden_size))
         self.gelu_impl = BloomGelu()                                                                                                                                                                                 
-        self.dense_4h_to_h = nn.Linear(4 * hidden_size, hidden_size)                                                                                                                                                 
+        self.dense_4h_to_h = nn.Linear(int(config.up_scale_factor * hidden_size), hidden_size)            
         self.hidden_dropout = config.hidden_dropout
```
