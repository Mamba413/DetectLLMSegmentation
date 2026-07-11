#!/usr/bin/env bash
# Copyright (c) Jin Zhu.
#

echo "$(date), Setup the environment ..."
set -e  # exit on error

# -------------------------
# paths
# -------------------------
exp_path=exp_vary_paralen
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path

# -------------------------
# common configs
# -------------------------
task="rewrite"
dataset="govreport"
M="grok-4-1-fast-non-reasoning"
k_list=(2 3 4 5 6)
paragraphs_per_half=6
power=2.0
cp_method=DP

local_python_path=~/miniconda3/envs/dgpt/bin/python
trained_model_path="scripts/FineTune/ckpt/"
aux_model="gemma-1b"
CACHE_DIR='../cache'

# # -------------------------
# # Step 1: Data generation (The data already generated in the data/ folder). To generate your data, you need to configure API of Grok 
# # -------------------------
# for k in "${k_list[@]}"; do
#   out_file="$data_path/${dataset}_${M}_${task}_k${k}"

#   if [ -f "${out_file}.raw_data.json" ]; then
#     echo "$(date), Skipping k=$k (already exists): ${out_file}.raw_data.json"
#   else
#     echo "$(date), Generating data: k=$k ..."
#     "$local_python_path" scripts/data_builder_paragraph.py \
#       --dataset "$dataset" \
#       --task "$task" \
#       --n_samples 30 \
#       --k "$k" \
#       --paragraphs_per_half "$paragraphs_per_half" \
#       --base_model_name "$M" \
#       --output_file "$out_file" \
#       --do_temperature \
#       --temperature 0.8 \
#       --cache_dir "$CACHE_DIR"
#   fi
# done

# -------------------------
# Step 2: Run detectors
# -------------------------
for k in "${k_list[@]}"; do
  echo "$(date), Running detectors: k=$k ..."

  eval_data_path="$data_path/${dataset}_${M}_${task}_k${k}"
  eval_result_path="$res_path/${dataset}_${M}_${task}_k${k}"

  python scripts/detector_texttiling.py \
    --eval_dataset ${eval_data_path} \
    --output_file ${eval_result_path}

  # ------------------------------------------------
  # sentence-wise prediction baselines
  # ------------------------------------------------

  python scripts/detector_naive_sp.py \
    --from_pretrained ${trained_model_path} \
    ${PHI_ARGS} \
    --eval_dataset ${eval_data_path} \
    --output_file ${eval_result_path}

  python scripts/detector_voting_sp.py \
    --from_pretrained ${trained_model_path} \
    ${PHI_ARGS} \
    --width 3 \
    --eval_dataset ${eval_data_path} \
    --output_file ${eval_result_path}

  # CP: DP + NFT phi, none weights, one known change point
  "$local_python_path" scripts/detector_value_cp.py \
    --from_pretrained "$trained_model_path" \
    --base_model "$aux_model" \
    --phi NFT \
    --cp_method ${cp_method} \
    --cp_num 1 \
    --weight_type none \
    --eval_dataset "$eval_data_path" \
    --output_file "$eval_result_path" \
    --cache_dir "$CACHE_DIR"

  # CP: DP + NFT phi, token weights, one known change point
  "$local_python_path" scripts/detector_value_cp.py \
    --from_pretrained "$trained_model_path" \
    --base_model "$aux_model" \
    --phi NFT \
    --cp_method ${cp_method} \
    --cp_num 1 \
    --weight_type ntokens \
    --power "$power" \
    --eval_dataset "$eval_data_path" \
    --output_file "$eval_result_path" \
    --cache_dir "$CACHE_DIR"
done
