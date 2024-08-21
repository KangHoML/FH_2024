# $1: --seed
# $2: seed number
# $3: --train_diag_path
# $4: filename of $S1

CUDA_VISIBLE_DEVICES="0" python3 train.py --swer_path './sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat' \
                                          --meta_path '../../Dataset/subtask3/mdata.wst.txt.2023.08.23' \
                                          --num_aug 5 \
                                          --threshold 0.7 \
                                          --batch_size 16 \
                                          --key_size 512 \
                                          --mem_size 16 \
                                          --hops 3 \
                                          --eval_node '[6000,6000,6000,200][2000,2000]' \
                                          --drop_prob 0.1 \
                                          --use_batch_norm False \
                                          --use_dropout True \
                                          --use_multimodal False \
                                          --use_cl True \
                                          --optimizer 'SGD' \
                                          --learning_rate 1e-5 \
                                          --weight_decay 1e-3 \
                                          --epoch 10 \
                                          --max_grad_norm 20.0 \
                                          --save_freq 2 \
                                          $1 $2 \
                                          $3 $4
