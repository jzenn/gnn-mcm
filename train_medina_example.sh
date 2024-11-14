#!/bin/bash

# This script trains for very few epochs on the dataset used by Medina et al. (2022).
# You can use this script to test whether the installed libraries work, but the
# resulting trained model will not be useful because real training requires a lot
# more training epochs. For real training, see example command line parameters in
# the files in the directory "hyperparameters".

python main.py --experiment_base_path "$(pwd)/experiments" \
--data_path "$(pwd)/data/medina_2022/medina_data.csv" \
--experiment_name medina-baseline \
--number_of_solutes \
156 \
--number_of_solvents \
262 \
--data \
Medina \
--split_dataset \
--split_sizes \
10%10%80% \
--exclude_solutes_solvents_not_present_in_train \
--num_workers \
4 \
--data_ensemble_id \
1 \
--model \
Diagonal-Gaussian-PMF-VI \
--diagonal_prior \
--data_likelihood_std \
0.15 \
--number_of_samples_for_expectation \
16 \
--dimensionality_of_embedding \
16 \
--predict_from_prior \
--get_point_estimate \
 \
--maximize_entropy \
--batch_size \
1000 \
--number_epochs \
10 \
--use_seed \
--seed \
1 \
--save_test_summary \
--clipping_schedule \
--clip_grad_value \
--max_grad \
1.0 \
--clipping_schedule_max_grad_factor \
10.0 \
--clipping_schedule_fraction \
0.1 \
--lr \
0.001 \
--use_lr_schedule \
--lr_schedule \
Cyclical \
--lr_scheduler_number_of_cycles \
2 \
--graph_model \
FiLM \
--graph_featurizer \
simple-atom \
--graph_activation \
ELU \
--graph_jumping_knowledge \
cat \
--graph_norm \
LayerNorm \
--graph_final_prediction_agg \
mean \
--graph_num_relations \
4 \
--graph_dropout \
0.1 \
--graph_featurizer_atom_embedding_dim \
16 \
--graph_number_of_layers \
6 \
--graph_message_agg \
add \
--graph_residual_every_layer \
-1 \
--graph_bias \
--graph_hidden_channels \
16