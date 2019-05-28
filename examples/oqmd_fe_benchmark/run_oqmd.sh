CGNN_HOME=../..
DATASET=${CGNN_HOME}/OQMD
python ${CGNN_HOME}/src/cgnn.py \
  --num_epochs 300 \
  --batch_size 512 \
  --lr 0.001 \
  --n_node_feat 89 \
  --n_hidden_feat 96 \
  --n_graph_feat 192 \
  --n_conv 4 \
  --n_fc 2 \
  --use_batch_norm \
  --full_pooling \
  --gated_pooling \
  --dataset_path ${DATASET} \
  --split_file ${DATASET}/split.json \
  --target_name formation_energy_per_atom \
  --milestones 250 \
  --gamma 0.1 \
  --weight_decay 1e-6 \

