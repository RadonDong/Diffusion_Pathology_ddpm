#!/bin/bash

# "stabilityai/stable-diffusion-2-1-base" 768
# "runwayml/stable-diffusion-v1-5" 512

export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="BRCA_Common_CDH1_formalin_withtumor_Old_mutation1/*.jpg"
export OUTPUT_DIR="BRCA_Common_CDH1_formalin_withtumor_Old_mutation1_lora"


accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="BRCA_Common_CDH1_formalin_withtumor_Old_mutation1" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --dataloader_num_workers=4 \
  --checkpointing_steps=1000 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=3000 \
  --validation_prompt="a photo of BRCA_Common_CDH1_formalin_withtumor_Old_mutation1" \
  --validation_epochs=1000 \
  --seed="0" \
  --push_to_hub \
  --hub_token="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" \