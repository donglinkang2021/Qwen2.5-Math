PROMPT_TYPE="qwen25-math-cot"

# Qwen2.5-Math-1.5B-Instruct
# export CUDA_VISIBLE_DEVICES="0"
# MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-1.5B-Instruct"
# bash eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
bash eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# /data1/linkdom/hf_models/Open-RS1
export CUDA_VISIBLE_DEVICES="0"
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