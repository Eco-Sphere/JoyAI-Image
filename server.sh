export ASCEND_RT_VISIBLE_DEVICES=8,9
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501

export COND_CACHE=0
export UNCOND_CACHE=0
export ATTENTION_CACHE=0

export ADALN_FUSE=1
export ROPE_FUSE=1

torchrun --nproc_per_node=2 --master_port=${MASTER_PORT} inference.py \
  --ckpt-root /home/w00853565/JoyAI-Image-Edit \
  --prompt "I want the human from the second figure to lovingly play with the dog, petting its fluffy, light tan and white coat while the puppy looks up with its dark, expressive eyes." \
  --image ../test2.png \
  --output outputs/result.png \
  --seed 123 \
  --steps 50 \
  --guidance-scale 4.0 \
  --basesize 1024 \
  --cfg-size 2 \
  --keep-image-size \
  --warmup 
