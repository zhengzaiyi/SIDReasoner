export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0



# Office_Products, Industrial_and_Scientific
for category in "Office_Products"; do
{
    train_file=$(ls -f ./data/Amazon/train/${category}*.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    echo ${train_file} ${eval_file} ${info_file} ${test_file}
    
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port 29519 \
            sft_reasoning_activation.py \
            --base_model output_dir/MiniOneRec-SFT_Qwen3-1.7B_Office_Products/checkpoint-477 \
            --batch_size 1024 \
            --micro_batch_size 8 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir output_dir/reasoning-activation_MiniOneRec_Qwen3-1.7B_${category} \
            --wandb_project MiniOneRec \
            --wandb_run_name reasoning-activation_MiniOneRec_Qwen3-1.7B_${category} \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path ./data/Amazon/index/${category}.index.json \
            --item_meta_path ./data/Amazon/index/${category}.item.json \
            --reasoning_train_file ./data/Amazon/index/${category}.integrated_narrative.csv \
            --train_new_token_embeddings_only False
} > logs/reasoning-activation_MiniOneRec_Qwen3-1.7B_${category}.txt 2>&1
done





