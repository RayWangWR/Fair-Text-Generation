CUDA_VISIBLE_DEVICES=0 python run_debias.py \
    --model_name_or_path gpt2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 