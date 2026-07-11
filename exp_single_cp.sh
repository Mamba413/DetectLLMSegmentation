#!/usr/bin/env bash
# Copyright (c) Jin Zhu.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

echo "$(date), Setup the environment ..."
set -e  # exit on error

# -------------------------
# paths
# -------------------------
exp_path=exp_single_cp
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path

# -------------------------
# common configs
# -------------------------
task="continuation"
Ms=('claude-haiku-4-5')
datasets=("squad" "xsum" "writing")
phis=("NFT")
cp_methods=('DP')
power_list=(2.0)
local_python_path=~/miniconda3/envs/dgpt/bin/python

# # preparing dataset
# for M in "${Ms[@]}"; do
#   for D in "${datasets[@]}"; do
#     temperature=0.8
#     if [[ "$M" == "gpt-5-mini" ]]; then
#       temperature=1.0
#     fi

#     if [ "$D" = "squad" ]; then
#       n_split=2
#     else
#       n_split=2
#     fi

#     echo "$(date), Preparing dataset ${D}_${M}_${task} ..."
#     if [[ "$M" == "claude-haiku-4-5" || "$M" == "gpt-4o" || "$M" == "gpt-5" || "$M" == "gpt-5-mini" || "$M" == "grok-4-1-fast-non-reasoning" ]]; then
#       "$local_python_path" scripts/data_builder_sentence.py \
#         --dataset "$D" \
#         --task "$task" \
#         --n_samples 100 \
#         --n_split "$n_split" \
#         --base_model_name "$M" \
#         --output_file "$data_path/${D}_${M}_${task}" \
#         --do_temperature \
#         --temperature "$temperature"
#     else
#       python scripts/data_builder_sentence.py \
#         --dataset "$D" \
#         --task "$task" \
#         --n_samples 100 \
#         --n_split "$n_split" \
#         --base_model_name "$M" \
#         --output_file "$data_path/${D}_${M}_${task}" \
#         --do_temperature \
#         --temperature "$temperature"
#     fi
#   done
# done

trained_model_path="scripts/FineTune/ckpt/"
aux_model="gemma-1b-instruct"

# -------------------------
# main experiment loop
# -------------------------
for M in "${Ms[@]}"; do
  echo "##################################################"
  echo "$(date), Model: $M"
  echo "##################################################"

  for D in "${datasets[@]}"; do
    eval_data_path="$data_path/${D}_${M}_${task}"
    eval_result_path="$res_path/${D}_${M}_${task}"

    # dataset-specific cp_num
    if [ "$D" = "squad" ]; then
      cp_num=1
    else
      cp_num=1
    fi

    echo "=================================================="
    echo "$(date), Dataset: $D (cp_num=$cp_num)"
    echo "=================================================="

    # ------------------------------------------------
    # texttiling baseline (no model needed, run once per dataset)
    # ------------------------------------------------
    python scripts/detector_texttiling.py \
      --eval_dataset ${eval_data_path} \
      --output_file ${eval_result_path}

    for phi in "${phis[@]}"; do
      echo "---- Phi: $phi ----"

      # phi-specific arguments
      PHI_ARGS=""
      if [ "$phi" = "Bino" ]; then
        PHI_ARGS="--phi Bino --aux_model ${aux_model}"
      elif [ "$phi" = "FDGPT" ]; then
        if [[ "$M" == "claude-haiku-4-5" || "$M" == "gpt-4o" || "$M" == "gpt-5" || "$M" == "gpt-5-mini" || "$M" == "grok-4-1-fast-non-reasoning" ]]; then
            PHI_ARGS="--phi FDGPT --base_model ${aux_model} --aux_model ${aux_model}"
        else
            PHI_ARGS="--phi FDGPT --base_model ${M} --aux_model ${M}"
        fi
      else
        PHI_ARGS="--phi FT"
      fi

      # detector_value_cp uses NFT when baselines use FT
      if [ "$phi" = "FT" ]; then
        CP_PHI_ARGS="--phi NFT"
      else
        CP_PHI_ARGS="$PHI_ARGS"
      fi

      if [[ "$M" == "claude-haiku-4-5" || "$M" == "gpt-4o" || "$M" == "gpt-5" || "$M" == "gpt-5-mini" || "$M" == "grok-4-1-fast-non-reasoning" ]]; then
          BASE_MODEL_ARGS="--base_model ${aux_model}"
      else
          BASE_MODEL_ARGS="--base_model ${M}"
      fi
      # ------------------------------------------------
      # sentence-wise prediction baselines
      # ------------------------------------------------
      python scripts/detector_llm_inquiry.py \
        ${BASE_MODEL_ARGS} \
        --eval_dataset ${eval_data_path} \
        --output_file ${eval_result_path} 

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

      # ------------------------------------------------
      # pald baseline (squad only — too slow for other datasets)
      # ------------------------------------------------
      if [ "$D" = "squad" ]; then
        if [ "$phi" = "FT" ]; then
          PALD_BASE_ARGS="--base_model gemma-1b"
        else
          PALD_BASE_ARGS="${BASE_MODEL_ARGS}"
        fi
        python scripts/detector_pald.py \
          --from_pretrained ${trained_model_path} \
          ${PHI_ARGS} \
          ${PALD_BASE_ARGS} \
          --eval_dataset ${eval_data_path} \
          --output_file ${eval_result_path}
      fi

      # ------------------------------------------------
      # change-point based methods
      # ------------------------------------------------
      for cp_method in "${cp_methods[@]}"; do
        echo "  >> CP method: $cp_method"

        # naive version
        python scripts/detector_value_cp.py \
          --from_pretrained ${trained_model_path} \
          ${CP_PHI_ARGS} \
          --cp_method ${cp_method} \
          --power 0.0 \
          --weight_type 'none' \
          --cp_selection 'aic' \
          --eval_dataset ${eval_data_path} \
          --output_file ${eval_result_path}

        # weighted version
        for power in "${power_list[@]}"; do
          python scripts/detector_value_cp.py \
            --from_pretrained ${trained_model_path} \
            ${CP_PHI_ARGS} \
            --cp_method ${cp_method} \
            --power ${power} \
            --cp_selection 'aic' \
            --weight_type 'invar' \
            --eval_dataset ${eval_data_path} \
            --output_file ${eval_result_path}
        done
      done
    done
  done
done

# /usr/bin/shutdown
