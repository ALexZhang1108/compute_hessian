#!/bin/bash

script_dir=$(dirname "$(realpath "$0")")

# 创建训练结果目录
mkdir -p $script_dir/trained_MLP

# 基本参数设置
batch_size=256
learning_rate=0.05
epochs=30
hidden_size=5
seed=0
weight_decay=0
momentum=0.9
save_epoch=0.05  # 每0.05个epoch保存一次检查点

# 需求：先完整训练一次（或部分训练）得到基线ckpt，然后在 0.5 / 1.0 / 2.0 epoch 恢复，
# 在不同恢复点进行 WSD decay，并在 decay 过程中按 save_epoch 频率存 ckpt

base_save_path=$script_dir/trained_MLP/bs${batch_size}_lr${learning_rate}_e${epochs}_hidden${hidden_size}_SGD_m${momentum}_wsd0_seed${seed}

echo "========================================================="
echo "预训练阶段：不使用 WSD，按 ${save_epoch} epoch 间隔保存 ckpt"
echo "========================================================="
mkdir -p $base_save_path
touch $base_save_path/pretrain.log

python $script_dir/single_layer_MLP_decaying_train.py \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --save_epoch $save_epoch \
    --hidden_size $hidden_size \
    --epochs $epochs \
    --save $script_dir/trained_MLP/ \
    --momentum $momentum \
    --weight_decay $weight_decay \
    --wsd_decay_steps 0 \
    --seed $seed 2>&1 | tee -a $base_save_path/pretrain.log

# 从 0.5 / 1.0 / 2.0 epoch 的 ckpt 恢复，进行 decay 训练
for resume_epoch in 0.500 1.000 2.000; do
  resume_dir=$script_dir/trained_MLP/bs${batch_size}_lr${learning_rate}_e${epochs}_hidden${hidden_size}_SGD_m${momentum}_wsd0_seed${seed}/ckpts/epoch_${resume_epoch}_step_
  # 自动找匹配该 epoch 的 wsd0 ckpt 目录
  match=$(ls -d ${resume_dir}* 2>/dev/null | head -n 1)
  if [ -z "$match" ]; then
    echo "未找到恢复点 epoch_${resume_epoch} 对应的 ckpt 目录，跳过该恢复点"
    continue
  fi
  resume_path=${match}/ckpt.pt
  if [ ! -f "$resume_path" ]; then
    echo "恢复文件不存在: $resume_path，跳过"
    continue
  fi

  # 针对每个恢复点，选择不同的 decay 时长，可按需调整
  # 示例：在每个恢复点都进行 0.5 个 epoch 的 WSD 衰减
  decay_len=0.5
  tag="resume_${resume_epoch}_decay_${decay_len}"
  out_dir=$script_dir/trained_MLP/bs${batch_size}_lr${learning_rate}_e${epochs}_hidden${hidden_size}_SGD_m${momentum}_wsd${decay_len}_${tag}_seed${seed}
  mkdir -p $out_dir
  touch $out_dir/train.log

  echo "========================================================="
  echo "从 epoch ${resume_epoch} 恢复，进行 WSD decay ${decay_len} 个 epoch，并按 ${save_epoch} 保存 ckpt"
  echo "恢复文件：$resume_path"
  echo "输出目录：$out_dir"
  echo "========================================================="

  python $script_dir/single_layer_MLP_decaying_train.py \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --save_epoch $save_epoch \
      --hidden_size $hidden_size \
      --epochs $epochs \
      --save $script_dir/trained_MLP/ \
      --momentum $momentum \
      --weight_decay $weight_decay \
      --wsd_decay_steps $decay_len \
      --seed $seed \
      --resume $resume_path 2>&1 | tee -a $out_dir/train.log
done

echo "流水线完成：预训练 + 多恢复点 decay 训练已结束"