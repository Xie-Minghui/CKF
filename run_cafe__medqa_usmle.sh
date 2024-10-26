#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
dt=`date '+%Y%m%d_%H%M%S'`


dataset="medqa_usmle"
#model='cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
model='SapBERT-from-PubMedBERT-fulltext'
shift
shift
args=$@


elr="5e-5"
dlr="1e-3"
bs=128
mbs=2
sl=512
n_epochs=50
ent_emb='ddb'
num_relation=34 #(15 +2) * 2: originally 15, add 2 relation types (QA context -> Q node; QA context -> A node), and double because we add reverse edges


k=5 #num of gnn layers
gnndim=200
unfrz=4


echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "enc_name: $model"
echo "batch_size: $bs"
echo "learning_rate: elr $elr dlr $dlr"
echo "gnn: dim $gnndim layer $k"
echo "******************************"

save_dir_pref='saved_models'
mkdir -p $save_dir_pref
mkdir -p logs

path_abs="xxx"  # your project

###### Training ######
for seed in 0; do
  python3 -u cafe.py --dataset $dataset \
      --encoder $model -k $k --gnn_dim $gnndim -elr $elr -dlr $dlr -bs $bs -mbs $mbs -sl $sl --seed $seed \
      --save_model \
      --num_relation $num_relation \
      --n_epochs $n_epochs --max_epochs_before_stop 50 --unfreeze_epoch $unfrz \
      --train_adj ${path_abs}/data/${dataset}/graph/train.graph.adj.pk \
      --dev_adj   ${path_abs}/data/${dataset}/graph/dev.graph.adj.pk \
      --test_adj  ${path_abs}/data/${dataset}/graph/test.graph.adj.pk \
      --train_statements  ${path_abs}/data/${dataset}/statement/train.statement.jsonl \
      --dev_statements  ${path_abs}/data/${dataset}/statement/dev.statement.jsonl \
      --test_statements  ${path_abs}/data/${dataset}/statement/test.statement.jsonl \
      --ent_emb ${ent_emb} \
      --save_dir ${save_dir_pref}/${dataset}/enc-sapbert__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt} $args \
  > logs/train_${dataset}__enc-sapbert__k${k}__gnndim${gnndim}__bs${bs}__sl${sl}__unfrz${unfrz}__seed${seed}__${dt}.log.txt
done

# --save_model \
# --save_dir ${save_dir_pref}/${dataset}/enc-sapbert__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt} $args \