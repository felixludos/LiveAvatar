CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=OFF

export ENABLE_COMPILE=false

# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nproc_per_node=1 --master_port=29101  minimal_inference/s2v_streaming_interact.py \
#      --ulysses_size 1 \
#      --task s2v-14B \
#      --size "384*256" \
#      --base_seed 420 \
#      --training_config liveavatar/configs/s2v_causal_sft.yaml \
#      --offload_model True \
#      --convert_model_dtype \
#      --prompt "A professional speaks confidently directly to the camera." \
#      --image "/mnt/c/Users/anwan/OneDrive/Khan/maity/vidLink/local_data/avatars/sales_executive/executive.png" \
#      --audio "/mnt/c/Users/anwan/OneDrive/Khan/maity/vidLink/video_generators/multitalk/local_data/sales_test/executive/s1.wav" \
#      --infer_frames 48 \
#      --load_lora \
#      --lora_path_dmd "Quark-Vision/Live-Avatar" \
#      --sample_steps 4 \
#      --sample_guide_scale 0 \
#      --num_clip 100 \
#      --num_gpus_dit 1 \
#      --sample_solver euler \
#      --single_gpu \
#      --ckpt_dir ckpt/Wan2.2-S2V-14B/ \
#      --fp8 \
#      --offload_kv_cache \
#      --t5_cpu
#      # --lora_path_dmd "Quark-Vision/Live-Avatar" \

IMAGE="/mnt/c/Users/anwan/OneDrive/Khan/maity/vidLink/local_data/avatars/sales_executive/executive.png"
AUDIO="/mnt/c/Users/anwan/OneDrive/Khan/maity/vidLink/video_generators/multitalk/local_data/sales_test/executive/s1.wav"
PROMPT="A professional speaks confidently directly to the camera."

IMAGE="/mnt/c/Users/anwan/OneDrive/Khan/maity/vidLink/local_data/avatars/greenscreen/green_woman0.png"
AUDIO="/mnt/c/Users/anwan/OneDrive/Khan/maity/vidLink/video_generators/multitalk/local_data/sales_test/executive/s1.wav"
PROMPT="A professional speaks confidently directly to the camera."

IMAGE="/mnt/c/Users/anwan/OneDrive/Khan/maity/vidLink/local_data/avatars/greenscreen/green_woman4.png"
AUDIO="/mnt/c/Users/anwan/OneDrive/Khan/maity/vidLink/video_generators/multitalk/local_data/sales_test/green_woman4/s1.wav"
PROMPT="A professional speaks confidently directly to the camera."


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nproc_per_node=1 --master_port=29101  minimal_inference/s2v_streaming_interact.py \
     --ulysses_size 1 \
     --task s2v-14B \
     --size "384*256" \
     --base_seed 420 \
     --training_config liveavatar/configs/s2v_causal_sft.yaml \
     --offload_model True \
     --convert_model_dtype \
     --prompt "$PROMPT" \
     --image "$IMAGE" \
     --audio "$AUDIO" \
     --infer_frames 48 \
     --load_lora \
     --lora_path_dmd "Quark-Vision/Live-Avatar" \
     --sample_steps 4 \
     --sample_guide_scale 0 \
     --num_clip 100 \
     --num_gpus_dit 1 \
     --sample_solver euler \
     --single_gpu \
     --ckpt_dir ckpt/Wan2.2-S2V-14B/ \
     --fp8 \
     --offload_kv_cache \
     --t5_cpu
     # --lora_path_dmd "Quark-Vision/Live-Avatar" \
     

