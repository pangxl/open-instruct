python scripts/submit_dpo_job.py \
  --default_beaker_config configs/beaker_configs/default_finetune.yaml \
  --config configs/train_configs/sft/safety_mix/llama-3.1-8b-dpo-pku.yaml\
  --cluster ai2/jupiter-cirrascale-2 \
  --priority high \
  --exp_name nd-DPO-tulu3-sft-wildgaurdmixtrain-mixv3.4-valmix-pkh \
  --num_gpus 8
