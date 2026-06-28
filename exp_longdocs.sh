#!/usr/bin/env bash
# Copyright (c) Jin Zhu.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
python_path=/Users/j.zhu.7@bham.ac.uk/miniconda3/envs/dgpt/bin/python
exp_path=exp_longdocs
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path

gpu_device='cuda'
M="gemini-2.5-flash"
dataset_files="squad_gemini-2.5-flash_polish writing_gemini-2.5-flash_expand"
sub_n_tokens="5 10 20 40 80 160 320"
M1="gemma-1b"
M2="gemma-1b-instruct"
D="squad"

# preparing dataset
for DF in $dataset_files; do
  for T in $sub_n_tokens; do
    echo date, Preparing dataset ${DF}_${T} ...
    python classifier/data_truncator.py \
      --input_file $data_path/${DF} \
      --output_file $data_path/${DF}_${T} \
      --max_words ${T}
  done
done

# evaluate various method
for DF in $dataset_files; do
    for T in $sub_n_tokens; do
        # evaluate AdaDetectGPT method
        trained_model_path=classifier/FineTune/ckpt/
        python classifier/detect_gpt_adajasa.py --base_model "$M1" --from_pretrained ${trained_model_path} --eval_model --eval_dataset $data_path/${DF}_${T} --output_file $res_path/${DF}_${T} --device $gpu_device

        # evaluate supervised detectors
        echo `date`, Evaluating OPENAI on ${DF}_${T}   ...
        SM="roberta-large-openai-detector"
        python classifier/supervised.py --model_name ${SM} --dataset ${D} --dataset_file $data_path/${DF}_${T} --output_file $res_path/${DF}_${T} --device $gpu_device

        # evaluate Fast-DetectGPT
        echo `date`, Evaluating Fast-DetectGPT on ${DF}_${T} with ${M1}_${M2} ...
        python classifier/fast_detect_gpt.py --sampling_model_name $M1 --scoring_model_name $M2 --dataset $D --dataset_file $data_path/${DF}_${T}  --output_file $res_path/${DF}_${T} --discrepancy_analytic --device $gpu_device

        # evaluate Binoculars
        echo `date`, Evaluating Binoculars on ${DF}_${T} with ${M1}_${M2} ...
        python classifier/detect_bino.py --model1_name $M1 --model2_name $M2 --dataset $D --dataset_file $data_path/${DF}_${T} --output_file $res_path/${DF}_${T} --device $gpu_device
    done
done
