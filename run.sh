export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

base_model_path="/path/to/models"

full_model_names=(
    "openchat-3___5-0106"
    "Mistral-7B-Instruct-v0___1"
    "Qwen2___5-7B-Instruct"
    "vicuna-7b-v1.5"
    "openchat-3.5-0106"
    "Meta-Llama-3-8B-Instruct"
    "h2ogpt-4096-llama2-7b-chat"
)


for full_model_name in ${full_model_names[@]}; do


model=${base_model_path}/${full_model_name}
model_name=$(basename ${full_model_name})

system_prompt_type="all"
HF_MODELS=/path/to/models

python estimate.py \
    --system_prompt_type ${system_prompt_type} \
    --config sampling --pretrained_model_path ${model}
# # echo """


prompt_length="default"

# echo """
python train_paralled.py \
    --system_prompt_type ${system_prompt_type} --prompt_length ${prompt_length} \
    --config sampling --pretrained_model_path ${model} \
    --use_paralled_multilingual 
# """

# done



languages=(
    "english"
    "danish"
    "korean"
    "greek"
    "irish"
)

for lang in ${languages[@]};do

# ######### none setting
### malicious

python generate.py \
    --use_sampling --n_samples 25 --use_malicious --pretrained_model_path ${model} \
    --dataset_language ${lang}  --prompt_length ${prompt_length} --system_prompt_type ${system_prompt_type}\
    --use_soft_prompt \
    --use_paralled_multilingual 

python evaluate.py \
    --config sampling --use_malicious --evaluator_path ${HF_MODELS}/Meta-Llama-Guard-2-8B \
    --model_names ${model_name}_with_soft_${system_prompt_type}_${prompt_length} \
    --dataset_language ${lang} \
    --use_paralled_multilingual
### advbench
python generate.py \
    --use_sampling --n_samples 25 --use_advbench --pretrained_model_path ${model} \
    --dataset_language ${lang} --prompt_length ${prompt_length}  --system_prompt_type ${system_prompt_type}\
    --use_soft_prompt \
    --use_paralled_multilingual 

python evaluate.py \
    --config sampling --use_advbench --evaluator_path ${HF_MODELS}/Meta-Llama-Guard-2-8B \
    --model_names ${model_name}_with_soft_${system_prompt_type}_${prompt_length} \
    --dataset_language ${lang} \
    --use_paralled_multilingual 
done
