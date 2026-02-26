CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=OFF

export PROMPT="A professional woman with long dark hair, wearing a dark blazer over a black top, stands in a brightly lit, modern open office. As she speaks directly to the camera, she begins with a small forward lean and an open hand gesture. Following a brief pause and a slight head tilt, she steps slightly to one side accompanied by a sweeping hand motion. She then gestures outward as if presenting a concept. After a small beat, she uses a subtle counting gesture with her fingers, finishing her delivery by establishing stronger eye contact and giving a firm, confident nod."

export IMAGE="/mnt/c/Users/anwan/OneDrive/Khan/maity/vidLink/local_data/avatars/outreach/avatar_side.jpeg"
export AUDIO="/mnt/c/Users/anwan/OneDrive/Khan/maity/vidLink/video_generators/multitalk/local_data/outreach_test/avatar_sid/s1.wav"

export ENABLE_COMPILE=false
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nproc_per_node=1 --master_port=29101  minimal_inference/s2v_streaming_interact.py \
     --ulysses_size 1 \
     --task s2v-14B \
     --size "704*384" \
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
     --sample_steps 8 \
     --sample_shift 3 \
     --sample_guide_scale 0 \
     --num_clip 50 \
     --num_gpus_dit 1 \
     --sample_solver euler \
     --single_gpu \
     --ckpt_dir ckpt/Wan2.2-S2V-14B/ \
     --fp8 \
     --offload_kv_cache \
     --t5_cpu
     # --lora_path_dmd "Quark-Vision/Live-Avatar" \

# --sample_solver unipc # unsupported - notimplementederror

     # --size "384*256" \
     
     
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
