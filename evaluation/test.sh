PROMPT_TYPE="qwen25-math-cot"

# My Model
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# BASE_MODEL_PATH="/data2/linkdom/converted_model_safetensors/rgrpo-qwen2.5-math-1.5b-instruct"
# CHECKPOINTS=(
#     "ckpt_000100"
#     "ckpt_000200"
#     "ckpt_000300"
#     "ckpt_000400"
#     "ckpt_000500"
#     "ckpt_000600"
#     "ckpt_000700"
#     "ckpt_000800"
#     "ckpt_000900"
#     "ckpt_001000"
#     "ckpt_001100"
#     "ckpt_001200"
# )

# for CKPT in "${CHECKPOINTS[@]}"; do
#     MODEL_NAME_OR_PATH="${BASE_MODEL_PATH}/${CKPT}"
#     echo "Evaluating ${MODEL_NAME_OR_PATH}"
#     bash eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH
# done

# Qwen2.5-Math-1.5B-Instruct
export CUDA_VISIBLE_DEVICES="0,1,2,3"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-1.5B-Instruct"
bash eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
export CUDA_VISIBLE_DEVICES="0,1,2,3"
MODEL_NAME_OR_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
bash eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# /data1/linkdom/hf_models/Open-RS1
export CUDA_VISIBLE_DEVICES="0,1,2,3"
MODEL_NAME_OR_PATH="/data1/linkdom/hf_models/Open-RS1"
bash eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# Qwen2.5-Math-7B-Instruct
export CUDA_VISIBLE_DEVICES="0,1,2,3"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-7B-Instruct"
bash eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
export CUDA_VISIBLE_DEVICES="0,1,2,3"
MODEL_NAME_OR_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
bash eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# GD-ML/Qwen2.5-Math-7B-GPG
export CUDA_VISIBLE_DEVICES="0,1,2,3"
MODEL_NAME_OR_PATH="GD-ML/Qwen2.5-Math-7B-GPG"
bash eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH
