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
# python train.py --log_path logs/v2_split_10ep/ \
#     --train_path "data/$FOLDER/gpt2_data/mem_dials_gpt2_train.json" \
#     --valid_path "data/$FOLDER/gpt2_data/mem_dials_gpt2_val.json" \
#     --special_tokens_path "data/$FOLDER/gpt2_data/mem_dials_gpt2_special_tokens.json" \
#     --feature_path "data/$FOLDER/memory_features/butd_10w_features" \
#     --train_batch_size 8 \
#     --predict_belief_state \
#     --n_epochs 10
    # --dataloader_dry_run

        # --test_set "data/$FOLDER/gpt2_data/mem_dials_gpt2_devtest.json" \
# CUDA_VISIBLE_DEVICES=1 \
#     python generate.py \
#         --model_checkpoint logs/v2_split_10ep/ \
#         --model_epoch 6 \
#         --test_set "data/$FOLDER/gpt2_data/mem_dials_gpt2_test.json" \
#         --special_tokens_path "data/$FOLDER/gpt2_data/mem_dials_gpt2_special_tokens.json" \
#         --feature_path "data/$FOLDER/memory_features/butd_10w_features" \
#         --output model_v2_ep6_test_results_regular.json


python utils/create_result_jsons.py \
    --memory_test_json "data/memory_dialog/final_data/new_split_feb22/mem_dials_test_v2.json" \
    --model_output_json "results/model_v2_ep6_test_results_regular.json"
    # --model_output_json "result_sample.json"
    


# DATA_PATH="data/$FOLDER/final_data"
# FEATURE_PATH="data/coco_butd_features_36/trainval_36/"
# FEATURE_PATH+="trainval_resnet101_faster_rcnn_genome_36.npy"
# FEATURE_PATH="data/visdial_img_feat.lmdb"

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

# FOLDER="data/memory_dialog/final_data/new_split_feb22"
# python utils/preprocess_memory_dataset.py \
#     --train_json_path "$FOLDER/mem_dials_train_v2.json" \
#     --unseen_json_path \
#         "$FOLDER/mem_dials_val_v2.json" \
#         "$FOLDER/mem_dials_test_v2.json" \
#     --save_folder "$FOLDER/../../gpt2_data/"

DATA_FOLDER="data/$FOLDER/final_data/new_split_feb22"
# python utils/analyze_memory_splits.py \
#     --input_train_json "$DATA_FOLDER/mem_dials_train_v2.json" \
#     --input_val_json "$DATA_FOLDER/mem_dials_val_v2.json" \
#     --input_test_json "$DATA_FOLDER/mem_dials_test_v2.json" \
#     --gpt_train_json "data/$FOLDER/gpt2_data/mem_dials_gpt2_train.json" \
#     --gpt_val_json "data/$FOLDER/gpt2_data/mem_dials_gpt2_val.json" \
#     --gpt_test_json "data/$FOLDER/gpt2_data/mem_dials_gpt2_test.json"


# FOLDER="memory_dialog"
# python utils/create_data_splits.py \
#     --input_json_path "data/$FOLDER/final_data/mem_dials_merged.json" \
#     --output_save_folder "data/$FOLDER/final_data/new_split_feb22/" \
#     --memory_graph_paths \
#         "data/$FOLDER/mscoco_memory_graphs_1k.json" \
#         "data/$FOLDER/memory_may21_v1_100graphs.json"
