CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=OFF

export ENABLE_COMPILE=false
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nproc_per_node=1 --master_port=29101  minimal_inference/s2v_streaming_interact.py \
     --ulysses_size 1 \
     --task s2v-14B \
     --size "384*256" \
     --base_seed 420 \
     --training_config liveavatar/configs/s2v_causal_sft.yaml \
     --offload_model True \
     --convert_model_dtype \
     --prompt "A confident woman speaks directly to the camera." \
     --image "/mnt/c/Users/anwan/OneDrive/Khan/maity/vidLink/local_data/avatars/sales_executive/executive.png" \
     --audio "/mnt/c/Users/anwan/OneDrive/Khan/maity/vidLink/video_generators/multitalk/local_data/sales_test/executive/s1.wav" \
     --infer_frames 48 \
     --load_lora \
     --lora_path_dmd "Quark-Vision/Live-Avatar" \
     --sample_steps 4 \
     --sample_guide_scale 10 \
     --num_clip 10 \
     --num_gpus_dit 1 \
     --sample_solver euler \
     --single_gpu \
     --ckpt_dir ckpt/Wan2.2-S2V-14B/ \
     --fp8 \
     --offload_kv_cache \
     --t5_cpu
     # --lora_path_dmd "Quark-Vision/Live-Avatar" \

# --sample_solver unipc # unsupported - notimplementederror

# /mnt/c/Users/anwan/OneDrive/Khan/maity/vidLink/video_generators/multitalk/local_data/business_test2/right/s1.wav
# /mnt/c/Users/anwan/OneDrive/Khan/maity/vidLink/local_data/avatars/business_man/right.png
# "A corporate video where a handsome, middle-aged man uncrosses his arms and begins speaking directly to the camera with a friendly smile, gesturing naturally as he speaks confidently and clearly, conveying a sense of professionalism and approachability."
# "A corporate video where a handsome, middle-aged man speaks directly to the camera with a friendly smile."

##

# /mnt/c/Users/anwan/OneDrive/Khan/maity/vidLink/video_generators/multitalk/local_data/sales_test/executive/s1.wav
# /mnt/c/Users/anwan/OneDrive/Khan/maity/vidLink/local_data/avatars/sales_executive/executive.png
# "A confident woman speaks directly to the camera."


     # --prompt "A stout, cheerful dwarf with a magnificent braided beard adorned with metal rings, wearing a heavy leather apron. He's standing in his fiery, cluttered forge, laughing heartily as he explains the mastery of his craft, holding up a glowing hammer. Style of Blizzard Entertainment cinematics (like World of Warcraft), warm, dynamic lighting from the forge."  \
     # --image "examples/dwarven_blacksmith.jpg" \
     # --audio "examples/dwarven_blacksmith.wav" \
