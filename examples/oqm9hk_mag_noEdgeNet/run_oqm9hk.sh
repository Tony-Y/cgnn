CGNN_HOME=../..
DATASET=${CGNN_HOME}/OQM9HK
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
  --n_postconv_net_layers 2 \
  --use_postconv_net_batch_norm \
  --dataset_path ${DATASET} \
  --split_file ${DATASET}/split.json \
  --target_name magnetization_per_atom \
  --milestones 300 \
  --gamma 1e-4 \
  --cosine_annealing \
  --weight_decay 1e-6 \
  --decoupled_wd \
  --summary_writer \
  --warmup \
  --output_activation softplus
