#!/bin/bash



# export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"

export OUTPUT_DIR=BRCA_Common_CDH1_formalin_withtumor_Old_mutation1_generator_06
#STEP=(2500 3000 3500)

#for step in ${STEP[@]};
#do
    #export LORA_NAME="kaneyxx/black_LUAD_2x_"$step"_1e-4"
    export LORA_NAME="BRCA_Common_CDH1_formalin_withtumor_Old_mutation1_lora/pytorch_lora_weights.bin"
    
    #echo "Current repo:" $LORA_NAME
    python image_generation.py \
      --pretrained_lora=$LORA_NAME  \
      --inference_prompt="a photo of BRCA_Common_CDH1_formalin_withtumor_Old_mutation1" \
      --output_dir=$OUTPUT_DIR \
      --sample_batch_size=5 \
      --sample_batch_count=2000 \
      --inference_steps=25
    #done