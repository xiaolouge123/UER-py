CUDA_VISIBLE_DEVICES=0 python3 run_classifier_cv.py --pretrained_model_path ~/pretrain_model/tencent/mixed_corpus_bert_xlarge_model.bin \
                          --config_path ~/UER-py/models/bert_xlarge_config.json \
                          --vocab_path models/google_zh_vocab.txt \
                          --train_path ~/ccf-2020/multi_cls/data/cls_10_data.csv \
                          --train_features_path ./stacking_features/tencent_xlarge_5kfold_featrues.txt \
                          --seq_length 512 \
                          --epochs_num 5 \
                          --batch_size 4 \
                          --test_run True \
                          --encoder bert