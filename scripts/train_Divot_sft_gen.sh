JOBNAME=${1:-'experiement'}

ps aux | grep 'src/train/train' | awk '{print $2}' | xargs kill -9
ps aux | grep 'debug' | awk '{print $2}' | xargs kill -9
set -x

PROJ_PATH=.
exp_name='sft_gen'
OUTPUT_PATH=train_outputs/${exp_name}

export PYTHONPATH=Divot/peft/src:$PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:Divot/src:

export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECKS_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=eth1
export UCX_NET_DEVICES=eth1
export NCCL_IB_HCA=mlx5
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1
export GLOO_SOCKET_IFNAME=eth1
export NCCL_DEBUG=info
mkdir -p $OUTPUT_PATH

NODE_RANK=$1
torchrun --nproc_per_node=8 --nnodes=2 --master_addr=11.220.5.202 --master_port=23457 --node_rank=$NODE_RANK \
    ${PROJ_PATH}/src/train/train_Divot.py \
    --image_transform ${PROJ_PATH}/configs/processer/dc_256_transform.yaml \
    --tokenizer ${PROJ_PATH}/configs/tokenizer/Divot_mistral_tokenizer_instruct.yaml \
    --visual_encoder ${PROJ_PATH}/configs/visual_encoder/Divot_video.yaml \
    --llm_model ${PROJ_PATH}/configs/clm_models/mistral7b_lora_pretrain.yaml \
    --agent_model ${PROJ_PATH}/configs/clm_models/agent_7b_in64_out64_video_gmm_pretrain_gen.yaml \
    --train_dataset ${PROJ_PATH}/configs/data/clm_sft_multi_data_video_gen.yaml \
    --output_dir ${OUTPUT_PATH} \
    --expr_name  ${exp_name} \
    --learning_rate 1e-4 \
    --batch_size 50 \
    --num_frames 5 \
    --weight_decay 0.05 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --gradient_accumulation_steps 1 \
    --mixed_precision bf16 \
    --num_train_epochs 200 \
    --max_steps 100000 \
    --save_steps 2000 \
    --lr_scheduler_type cosine \
    --warmup_steps 500 \
    --min_lr_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed_plugin ${PROJ_PATH}/configs/accelerate/deepspeed_stage_2_offload.yaml \


echo '--------------------------'
echo main training task done
echo '--------------------------'
