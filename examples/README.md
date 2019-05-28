# Examples

The benchmark, no-EdgeNet, and complete models, which are described in the CGNN paper, are presented below as examples.

## Benchmark Model
The directory `oqmd_fe_benchmark` contains a shell script that trains the benchmark model for the formation energy prediction.

**Model Definition**
```
--n_hidden_feat 96 \
--n_graph_feat 192 \
--n_conv 4 \
--n_fc 2 \
--use_batch_norm \
--full_pooling \
--gated_pooling \
```

You can train this model as follows:
```
cd oqmd_fe_benchmark
bash run_oqmd.sh >& log_oqmd &
```

## No EdgeNet Model
The directory `oqmd_fe_noEdgeNet` contains a shell script that trains the CGNN model without the EdgeNet for the formation energy prediction.

**Model Definition**
```
--n_hidden_feat 96 \
--n_graph_feat 192 \
--n_conv 4 \
--n_fc 2 \
--use_batch_norm \
--full_pooling \
--gated_pooling \
--n_postconv_net_layers 2 \
--use_postconv_net_batch_norm \
```

You can train this model as follows:
```
cd oqmd_fe_noEdgeNet
bash run_oqmd.sh >& log_oqmd &
```

The directories `oqmd_vol_noEdgeNet`, `oqmd_bg_noEdgeNet`, and `oqmd_mag_noEdgeNet` also contain a shell script for the volume, band gap, and total magnetization predictions, respectively.

## Complete Model
The directory `oqmd_fe_complete` contains a shell script that trains the complete CGNN model for the formation energy prediction.

**Model Definition**
```
--n_hidden_feat 96 \
--n_graph_feat 192 \
--n_conv 4 \
--n_fc 2 \
--use_batch_norm \
--full_pooling \
--gated_pooling \
--n_postconv_net_layers 2 \
--use_postconv_net_batch_norm \
--n_edge_net_feat 144 \
--n_edge_net_layers 1 \
--use_fast_edge_network \
--fast_edge_network_type 1 \
--use_edge_net_shortcut \
```

You can train this model as follows:
```
cd oqmd_fe_complete
bash run_oqmd.sh >& log_oqmd &
```

## Results

**Training histories**

![Training histories](../figs/fig_training_histories.png)

**Testing errors for the formation energy predictions, and training speeds and times**

| Model      | RMSE (meV) | MAE (meV)  | Speed (examples/sec) | Time   |
|------------|-----------:|-----------:|---------------------:|-------:|
| Benchmark  |       86.8 |       43.5 |                13868 | 2h 42m |
| No EdgeNet |       83.8 |       41.1 |                11372 | 3h 18m |
| Complete   |       82.8 |       36.7 |                 5490 | 6h 50m |
| Ensemble   |       77.0 |       34.4 |                   NA |     NA |
| Database   |      124.2 |       84.8 |                   NA |     NA |

The benchmark, no-EdgeNet, and complete models were trained on a single GeForce GTX 1080 GPU. The ensemble prediction is the average of three predictions computed by those  models. The database error is the formation energy difference between the OQMD and the Materials Project entry, as described in the CGNN paper. The formation energy errors are expressed in milli-electron volts (meV).

**Testing errors for the volume deviation predictions**

| Model      | RMSE   | MAE    |
|------------|-------:|-------:|
| No EdgeNet | 0.0319 | 0.0175 |
| Database   | 0.0421 | 0.0270 |

The volume deviation is defined by 1 - Va/Vc, where Va denotes the total atomic volume, and Vc denotes the cell volume.

**Testing errors for the band gap predictions**

| Model      | RMSE   | MAE    |
|------------|-------:|-------:|
| No EdgeNet | 0.2602 | 0.0502 |
| Database   | 0.5288 | 0.1806 |

The band gap errors are expressed in electron volts.

**Testing errors for the total magnetization predictions**

| Model      | RMSE   | MAE    |
|------------|-------:|-------:|
| No EdgeNet | 0.1978 | 0.0826 |
| Database   | 0.4003 | 0.0938 |

The total magnetization errors are expressed in Bohr magnetons per atom.

Note that we could obtain results slightly different from the paper's ones, especially in the RMSE metric, because the splitting of the OQMD dataset differs between this repository and the CGNN paper.

(c) 2019 Takenori Yamamoto
