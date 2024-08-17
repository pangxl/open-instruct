#!/bin/bash

mkdir -p output/shards
num_prompts=100000
num_shards=100
prompts_per_shard=$((num_prompts / num_shards))
shared_hf_repo_id=rejection_sampling_$RANDOM 
num_completions=16
generation_model=/model
reward_model=allenai/llama-3-tulu-2-8b-uf-mean-rm
sft_dataset=ai2-adapt-dev/metamathqa-math-reformatted
on_jupyter=true
num_gpus=1
mkdir -p output/shards/$shared_hf_repo_id

# Prepare the command string
command=""

# Loop through shards
for ((i=0; i<num_shards; i++))
do
    # Calculate start and end indices for this shard
    start_idx=$((i * prompts_per_shard))
    end_idx=$(((i + 1) * prompts_per_shard))
    
    # Adjust the end index for the last shard to include any remaining prompts
    if [ $i -eq $((num_shards - 1)) ]; then
        end_idx=$num_prompts
    fi
    
    # Build the command string for this shard
    shard_command="python open_instruct/rejection_sampling/generation.py \
    --dataset_name $sft_dataset \
    --model_name_or_path $generation_model \
    --dataset_start_idx $start_idx \
    --dataset_end_idx $end_idx \
    --save_filename output/shards/$shared_hf_repo_id/$i.jsonl \
    --num_completions $num_completions --tensor_parallel_size $num_gpus && \
    python open_instruct/rejection_sampling/rejection_sampling.py \
    --input_filename output/shards/$shared_hf_repo_id/$i.jsonl \
    --model_names_or_paths $reward_model \
    --save_filename output/shards/$shared_hf_repo_id/scores_$i.jsonl \
    --hf_repo_id $shared_hf_repo_id \
    --no_add_timestamp \
    --num_completions $num_completions \
    --push_to_hub \
    --num_gpus $num_gpus && \
    echo Finished shard $((i+1)) of $num_shards"

    # Add the shard command to the main command string
    if [ -z "$command" ]; then
        command="$shard_command"
    else
        command="$command -- $shard_command"
    fi
done

echo $command

# Run the combined command
echo "Submitting all shards in one command"
# if running on juptyer, use the following command

if [ "$on_jupyter" = true ]; then
    python mason.py \
        --cluster ai2/jupiter-cirrascale-2 \
        --image costah/open_instruct \
        --pure_docker_mode \
        --priority low \
        --preemptible \
        --no_mount_nfs --no_hf_cache_env \
        --budget ai2/allennlp \
        --beaker_dataset /model:hamishivi/llama-3-8b-tulu_v2_numina-WEIGHTED_MERGE-_dpo_norm_beta5_uf_1ep \
        --gpus $num_gpus -- $command
else
    echo "Running on Mason"
    python mason.py \
    --cluster ai2/allennlp-cirrascale ai2/pluto-cirrascale ai2/prior-cirrascale ai2/s2-cirrascale \
    --image costah/open_instruct \
    --pure_docker_mode \
    --priority low \
    --preemptible \
    --budget ai2/allennlp \
    --gpus $num_gpus -- $command
fi

echo "All shards submitted"