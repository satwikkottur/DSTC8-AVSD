FOLDER="avsd8"

# Original AVSD8 dataset.
# CUDA_VISIBLE_DEVICES=1 python train.py --log_path log \
#     --train_path "data/$FOLDER/train_set4DSTC7-AVSD.json" \
#     --valid_path "data/$FOLDER/valid_set4DSTC7-AVSD.json" \
#     --fea_path "data/$FOLDER/"\
#     --train_batch_size 2

FOLDER="memory_dialog"

# Memory Dialog.
# CUDA_VISIBLE_DEVICES=1 \
# python train.py --log_path logs/memory_train_v1_50ep/ \
#     --train_path "data/$FOLDER/gpt2_data/mem_dials_gpt2_train.json" \
#     --valid_path "data/$FOLDER/gpt2_data/mem_dials_gpt2_dev.json" \
#     --special_tokens_path "data/$FOLDER/gpt2_data/mem_dials_gpt2_special_tokens.json" \
#     --feature_path "data/$FOLDER/memory_features/butd_10w_features" \
#     --train_batch_size 8 \
#     --predict_belief_state \
#     --n_epochs 10
    # --dataloader_dry_run

        # --test_set "data/$FOLDER/gpt2_data/mem_dials_gpt2_devtest.json" \
# CUDA_VISIBLE_DEVICES=1 \
#     python generate.py \
#         --model_checkpoint logs/memory_train_v1/ \
#         --model_epoch 6 \
#         --test_set "data/$FOLDER/gpt2_data/mem_dials_gpt2_devtest.json" \
#         --special_tokens_path "data/$FOLDER/gpt2_data/mem_dials_gpt2_special_tokens.json" \
#         --feature_path "data/$FOLDER/memory_features/butd_10w_features" \
#         --output model_v1_ep6_devtest_results.json \


# python utils/create_result_jsons.py \
#     --memory_test_json "data/memory_dialog/final_data/mem_dials_devtest.json" \
#     --model_output_json "results/model_v1_ep6_devtest_results.json"
    # --model_output_json "result_sample.json"
    


DATA_PATH="data/$FOLDER/final_data"
# FEATURE_PATH="data/coco_butd_features_36/trainval_36/"
# FEATURE_PATH+="trainval_resnet101_faster_rcnn_genome_36.npy"
FEATURE_PATH="data/visdial_img_feat.lmdb"

# python utils/extract_memory_features.py \
#     --input_dialog_json data/$FOLDER/final_data/mem_dials_merged.json \
#     --input_memory_json \
#         data/$FOLDER/memory_may21_v1_100graphs.json \
#         data/$FOLDER/mscoco_memory_graphs_1k.json \
#     --input_feature_path $FEATURE_PATH \
#     --max_bboxes 10 \
#     --feature_save_path data/$FOLDER/memory_features/butd_10w_features/
    # --input_feature_path sample_features_butd.npy \


# CUDA_VISIBLE_DEVICES=1 python train.py --log_path log/
#     --train_path "data/$FOLDER/train_set4DSTC7-AVSD.json" \
#     --valid_path "data/$FOLDER/valid_set4DSTC7-AVSD.json" \
#     --fea_path "data/$FOLDER"

# python utils/preprocess_memory_dataset.py \
#     --train_json_path "data/$FOLDER/final_data/mem_dials_train.json" \
#     --unseen_json_path \
#         "data/$FOLDER/final_data/mem_dials_dev.json" \
#         "data/$FOLDER/final_data/mem_dials_devtest.json" \
#     --save_folder "data/$FOLDER/gpt2_data/"

python utils/analyze_memory_splits.py \
    --input_train_json "data/$FOLDER/final_data/mem_dials_train.json" \
    --input_dev_json "data/$FOLDER/final_data/mem_dials_dev.json" \
    --input_devtest_json "data/$FOLDER/final_data/mem_dials_devtest.json" \
    --input_test_json "data/$FOLDER/final_data/mem_dials_test.json" \
    --gpt_train_json "data/$FOLDER/gpt2_data/mem_dials_gpt2_train.json" \
    --gpt_dev_json "data/$FOLDER/gpt2_data/mem_dials_gpt2_dev.json" \
    --gpt_devtest_json "data/$FOLDER/gpt2_data/mem_dials_gpt2_devtest.json"
