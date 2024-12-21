python /home/featurize/transformers/examples/pytorch/image-classification/run_image_classification.py \
    --model_name_or_path microsoft/resnet-50 \
    --dataset_name wellCh4n/tomato-leaf-disease-image \
    --output_dir ./tomato_outputs_res_50/ \
    --remove_unused_columns False \
    --label_column_name label \
    --do_train \
    --do_eval \
    --learning_rate 2e-5 \
    --num_train_epochs 100 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337 \
    --ignore_mismatched_sizes True \
    --report_to tensorboard