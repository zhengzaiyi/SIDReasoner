
{
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0

# export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
for category in "Office_Products"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    echo ${train_file} ${eval_file} ${info_file} ${test_file}
    
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 \
            --master_port 12340 \
            sft_freeze_Qwen3.py \
            --base_model /home/yingzhi/huggingface_data/hub/Qwen3-1.7B \
            --batch_size 1024 \
            --micro_batch_size 1 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir output_dir/7TaskFull-E2E-GeneralReason_Qwen3-1.7B_${category} \
            --wandb_project MiniOneRec \
            --wandb_run_name 7TaskFull-E2E-GeneralReason_Qwen3-1.7B_${category} \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path ./data/Amazon/index/${category}.index.json \
            --item_meta_path ./data/Amazon/index/${category}.item.json \
            --llm_generated_data_path ./data/Amazon/index/${category}.item_enhanced_v2.json \
            --llm_generated_sequence_path ./data/Amazon/index/${category}.integrated_narrative.csv \
            --general_reasoning_path ./data/Amazon/general/sampled_data.arrow \
            --mask_assistant True
done
} > logs/7TaskFull-E2E-GeneralReason_Qwen3-1.7B_${category}.txt 2>&1
