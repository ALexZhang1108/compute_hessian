#!/bin/bash
set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")

# 默认参数（可用 getopts 覆盖）
batch_size=256
learning_rate=0.05
epochs=30
hidden_size=5
seed=0
momentum=0.9
save_epoch=0.05
decay_len=3
resume_epochs="28.000" # 逗号分隔的小数 epoch 进度
continue_after_wsd=true
out_base_dir="$script_dir/trained_MLP"
post_wsd_lr=1e-6

usage() {
  cat <<EOF
用法: $(basename "$0") [选项]

从预训练 ckpt（wsd0/0.0）按指定 epoch 恢复，执行 WSD decay 并保存 ckpt。

选项:
  -b <int>     batch_size (默认: $batch_size)
  -l <float>   learning_rate (默认: $learning_rate)
  -e <int>     epochs (总训练 epoch, 默认: $epochs)
  -H <int>     hidden_size (默认: $hidden_size)
  -s <int>     seed (默认: $seed)
  -m <float>   momentum (默认: $momentum)
  -S <float>   save_epoch (保存频率, <1 表示按 epoch 小数频率, 默认: $save_epoch)
  -d <float>   decay_len (WSD decay 的时长, 单位: epoch, 默认: $decay_len)
  -r <list>    恢复的 epoch 列表(逗号分隔), 例如: "0.500,1.000,2.000"
  -c           decay 完成后继续训练至总 epochs（传递 --continue_after_wsd）
  -p <float>   post_wsd_lr (decay 完成后固定学习率, 默认: $post_wsd_lr)
  -o <path>    输出根目录 (默认: $out_base_dir)
  -h           显示此帮助

示例:
  $(basename "$0") -b 256 -l 0.05 -e 30 -H 5 -s 0 -m 0.9 -S 0.05 -d 0.5 -r "0.500,1.000,2.000"
EOF
}

while getopts "b:l:e:H:s:m:S:d:r:cp:o:h" opt; do
  case "$opt" in
    b) batch_size="$OPTARG" ;;
    l) learning_rate="$OPTARG" ;;
    e) epochs="$OPTARG" ;;
    H) hidden_size="$OPTARG" ;;
    s) seed="$OPTARG" ;;
    m) momentum="$OPTARG" ;;
    S) save_epoch="$OPTARG" ;;
    d) decay_len="$OPTARG" ;;
    r) resume_epochs="$OPTARG" ;;
    c) continue_after_wsd=true ;;
    p) post_wsd_lr="$OPTARG" ;;
    o) out_base_dir="$OPTARG" ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done

python_script="$script_dir/single_layer_MLP_decaying_train.py"
if [ ! -f "$python_script" ]; then
  echo "找不到训练脚本: $python_script" >&2
  exit 1
fi

# 匹配预训练目录（兼容 wsd0 与 wsd0.0），按时间倒序优先
read -r -a pretrain_dirs <<< "$(ls -dt ${out_base_dir}/bs${batch_size}_lr${learning_rate}_e${epochs}_hidden${hidden_size}_SGD_m${momentum}_wsd0*_seed${seed} 2>/dev/null || true)"
if [ ${#pretrain_dirs[@]} -eq 0 ]; then
  echo "未找到预训练目录：${out_base_dir}/bs${batch_size}_lr${learning_rate}_e${epochs}_hidden${hidden_size}_SGD_m${momentum}_wsd0*__seed${seed}" >&2
  exit 1
fi

echo "候选预训练目录（按新到旧）："
for d in "${pretrain_dirs[@]}"; do echo "  - $d"; done

# 遍历需要恢复的 epoch 进度列表
IFS=',' read -r -a resume_list <<< "$resume_epochs"
for resume_epoch in "${resume_list[@]}"; do
  resume_epoch_trimmed="$(echo "$resume_epoch" | xargs)"
  [ -z "$resume_epoch_trimmed" ] && continue

  # 在所有候选预训练目录中查找对应的 ckpt 目录
  match=""
  for base in "${pretrain_dirs[@]}"; do
    hit=$(ls -d ${base}/ckpts/epoch_${resume_epoch_trimmed}_step_* 2>/dev/null | head -n 1 || true)
    if [ -n "$hit" ]; then
      match="$hit"
      pretrain_dir="$base"
      break
    fi
  done
  if [ -z "$match" ]; then
    echo "未找到恢复点 epoch_${resume_epoch_trimmed} 对应的 ckpt 目录，跳过该恢复点"
    continue
  fi
  resume_path="${match}/ckpt.pt"
  if [ ! -f "$resume_path" ]; then
    echo "恢复文件不存在: $resume_path，跳过"
    continue
  fi

  echo "找到恢复文件：$resume_path"

  tag="resume_${resume_epoch_trimmed}_decay_${decay_len}"
  # 与 Python 内部保存目录命名保持一致：..._wsd{decay}_seed{seed}_{tag}
  out_dir="${out_base_dir}/bs${batch_size}_lr${learning_rate}_e${epochs}_hidden${hidden_size}_SGD_m${momentum}_wsd${decay_len}_seed${seed}_${tag}"
  mkdir -p "$out_dir"
  touch "$out_dir/train.log"

  echo "========================================================="
  echo "从 epoch ${resume_epoch_trimmed} 恢复，进行 WSD decay ${decay_len} 个 epoch，并按 ${save_epoch} 保存 ckpt"
  echo "恢复文件：$resume_path"
  echo "输出目录：$out_dir"
  echo "========================================================="

  extra_flags=()
  if [ "$continue_after_wsd" = true ]; then
    extra_flags+=("--continue_after_wsd")
  fi

  python "$python_script" \
    --batch_size "$batch_size" \
    --learning_rate "$learning_rate" \
    --save_epoch "$save_epoch" \
    --hidden_size "$hidden_size" \
    --epochs "$epochs" \
    --save "$out_base_dir/" \
    --momentum "$momentum" \
    --weight_decay 0 \
    --wsd_decay_steps "$decay_len" \
    --seed "$seed" \
    --resume "$resume_path" \
    --run_tag "$tag" \
    --post_wsd_lr "$post_wsd_lr" \
    "${extra_flags[@]}" 2>&1 | tee -a "$out_dir/train.log"
done

echo "完成：从指定恢复点执行 WSD decay 并保存 ckpt 全部结束"


