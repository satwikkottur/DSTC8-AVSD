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
# python train.py --log_path log \
#     --train_path "data/$FOLDER/final_data/mem_dials_dev.json" \
#     --valid_path "data/$FOLDER/final_data/mem_dials_dev.json" \
#     --fea_path "data/$FOLDER/"\
#     --train_batch_size 2 \
#     --video_agnostic


DATA_PATH="data/$FOLDER/final_data"
FEATURE_PATH="data/coco_butd_features_36/trainval_36/"
FEATURE_PATH+="trainval_resnet101_faster_rcnn_genome_36.npy"

python utils/extract_memory_features.py \
    --input_dialog_json data/$FOLDER/final_data/mem_dials_merged.json \
    --input_memory_json \
        data/$FOLDER/memory_may21_v1_100graphs.json \
        data/$FOLDER/mscoco_memory_graphs_1k.json \
    --input_feature_path $FEATURE_PATH \
    --feature_save_path data/$FOLDER/memory_features


# CUDA_VISIBLE_DEVICES=1 python train.py --log_path log/
#     --train_path "data/$FOLDER/train_set4DSTC7-AVSD.json" \
#     --valid_path "data/$FOLDER/valid_set4DSTC7-AVSD.json" \
#     --fea_path "data/$FOLDER"
