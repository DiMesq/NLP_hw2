#!/bin/bash
source /home/dam740/python3-venv/bin/activate

# cnn
python train.py cnn /scratch/dam740/nlp_hw2/cnn_models 150 cat &> logs/cnn_150_cat.out &
python train.py cnn /scratch/dam740/nlp_hw2/cnn_models 300 cat &> logs/cnn_300_cat.out &
#python train.py cnn /scratch/dam740/nlp_hw2/cnn_models 500 cat &> logs/cnn_500_cat.out &
python train.py cnn /scratch/dam740/nlp_hw2/cnn_models 1000 cat &> logs/cnn_1000_cat.out &

#python train.py cnn /scratch/dam740/nlp_hw2/cnn_models 150 element_wise_mult &> logs/cnn_150_ewise.out &
#python train.py cnn /scratch/dam740/nlp_hw2/cnn_models 300 element_wise_mult &> logs/cnn_300_ewise.out &
python train.py cnn /scratch/dam740/nlp_hw2/cnn_models 500 element_wise_mult &> logs/cnn_500_ewise.out &
python train.py cnn /scratch/dam740/nlp_hw2/cnn_models 1000 element_wise_mult &> logs/cnn_1000_ewise.out &

# rnn
# python train.py rnn /scratch/dam740/nlp_hw2/rnn_models 150 cat &> logs/rnn_150_cat.out &
# python train.py rnn /scratch/dam740/nlp_hw2/rnn_models 300 cat &> logs/rnn_300_cat.out &
# python train.py rnn /scratch/dam740/nlp_hw2/rnn_models 500 cat &> logs/rnn_500_cat.out &
# python train.py rnn /scratch/dam740/nlp_hw2/rnn_models 1000 cat &> logs/rnn_1000_cat.out &

# python train.py rnn /scratch/dam740/nlp_hw2/rnn_models 150 element_wise_mult &> logs/rnn_150_ewise.out &
# python train.py rnn /scratch/dam740/nlp_hw2/rnn_models 300 element_wise_mult &> logs/rnn_300_ewise.out &
# python train.py rnn /scratch/dam740/nlp_hw2/rnn_models 500 element_wise_mult &> logs/rnn_500_ewise.out &
# python train.py rnn /scratch/dam740/nlp_hw2/rnn_models 1000 element_wise_mult &> logs/rnn_1000_ewise.out &


