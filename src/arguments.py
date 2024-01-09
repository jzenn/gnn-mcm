import argparse

data_choices = ["DDB", "Medina"]
mf_choices = ["Diagonal-Gaussian-PMF-VI", "No-Prior-MF"]
graph_choices = ["FiLM", "Sum-Formula"]
graph_featurizer_choices = ["sum-formula", "simple-atom"]
graph_norm_choices = ["BatchNorm", "LayerNorm", "none"]
graph_activation_choices = ["LeakyReLU", "ReLU", "tanh", "ELU"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Framework for training (Graph)-((P)MF) models on DDB/QM9/Toy."
    )
    parser.add_argument(
        "--slurm_job_id",
        type=int,
        help="Job-ID the job has on a slurm cluster.",
    )
    parser.add_argument(
        "--slurm_job_partition",
        type=str,
        help="Partition the job uses on a slurm cluster.",
    )
    parser.add_argument(
        "--slurm_job_name",
        type=str,
        help="Name the job has on a slurm cluster.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers to use for DataLoader.",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Number of samples loaded in advance by each DataLoader worker.",
    )
    parser.add_argument(
        "--load_from_path",
        type=str,
        help="Base path to load the model (and trainer) from "
        "(contains args.txt and checkpoints/).",
    )
    parser.add_argument(
        "--load_model_from_path",
        type=str,
        help="Base path to load the model from (contains args.txt and checkpoints/).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The model to use.",
        choices=mf_choices + graph_choices,
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="The data to use.",
        choices=data_choices,
    )
    parser.add_argument(
        "--data_ensemble_id",
        type=int,
        default=1,
        help="The ensemble data to train on (only applies to Medina data).",
    )
    parser.add_argument(
        "--number_of_distinct_atoms",
        type=int,
        default=12,
        help="The number of distinct atoms in the dataset.",
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="The path to the data."
    )
    parser.add_argument(
        "--split_dataset", action="store_true", help="Whether to split the dataset."
    )
    parser.add_argument(
        "--split_sizes",
        type=str,
        help="%-separated string of percentages (val%test%train%).",
    )
    parser.add_argument(
        "--do_not_validate",
        action="store_true",
        help="Whether to *not* validate (although splitting the dataset if "
        "--split_sizes might be set accordingly.",
    )
    parser.add_argument(
        "--validate_pmf_with_mse",
        action="store_true",
        help="Whether to validate a PMF model with MSE-loss instead of ELBO-loss",
    )
    parser.add_argument(
        "--random_zero_shot_prediction_from_rows_and_cols",
        action="store_true",
        help="Whether to randomly drop entries of the dataset for later "
        "zero-shot prediction.",
    )
    parser.add_argument(
        "--n_solutes_for_zero_shot",
        type=int,
        default=0,
        help="Number of solutes excluded from training for zero-shot prediction.",
    )
    parser.add_argument(
        "--n_solvents_for_zero_shot",
        type=int,
        default=0,
        help="Number of solvents excluded from training for zero-shot prediction.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Use WandB logging.",
    )
    parser.add_argument(
        "--wandb_initialized",
        type=bool,
        default=False,
        help="Internal parameter indicating whether wandb has already been intialized.",
    )
    parser.add_argument(
        "--wandb_file",
        type=str,
        help="Path to file specifying directories and API-key.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="chem-prior",
        help="WandB project name.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="jzenn",
        help="WandB entity name.",
    )
    parser.add_argument(
        "--wandb_log_frequency",
        type=int,
        default=500,
        help="WandB frequency to log parameter values and gradients.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="The name of the current experiment.",
    )
    parser.add_argument(
        "--experiment_base_path",
        type=str,
        required=True,
        help="The path where the experiment will be written to.",
    )
    parser.add_argument(
        "--save_test_summary",
        action="store_true",
        help="Whether to plot save the test-summary after training.",
    )
    parser.add_argument(
        "--get_point_estimate",
        action="store_true",
        help="Whether to get a point estimate after training.",
    )
    parser.add_argument(
        "--exclude_solutes_solvents_not_present_in_train",
        action="store_true",
        help="Whether to exclude solutes and solvents from testing that are "
        "not present in the training set.",
    )
    parser.add_argument(
        "--clip_grad_norm",
        action="store_true",
        help="Whether to clip the gradients norm of the model to --max_grad.",
    )
    parser.add_argument(
        "--clip_grad_value",
        action="store_true",
        help="Whether to clip the gradients of the model to "
        "(- --max_grad, + --max_grad).",
    )
    parser.add_argument(
        "--clipping_schedule",
        action="store_true",
        help="Whether to use a schedule in clipping the gradients "
        "(depending on epoch).",
    )
    parser.add_argument(
        "--clipping_schedule_fraction",
        type=float,
        default=0.1,
        help="Fraction of training epochs after clipping ends.",
    )
    parser.add_argument(
        "--clipping_schedule_max_grad_factor",
        type=float,
        default=5.0,
        help="Clipping is started at max_grad and increased to max_grad * factor.",
    )
    parser.add_argument(
        "--max_grad",
        type=float,
        default=5.0,
        help="To which norm/value the gradients are clipped when using "
        "--clip_gradients.",
    )
    parser.add_argument(
        "--number_of_solutes",
        type=int,
        help="The number of solutes.",
    )
    parser.add_argument(
        "--number_of_solvents",
        type=int,
        help="The number of solvents.",
    )
    parser.add_argument(
        "--dimensionality_of_embedding",
        type=int,
        help="Dimensionality of the embedding.",
    )
    parser.add_argument(
        "--maximize_entropy",
        action="store_true",
        help="Whether to maximize the variational entropy instead of "
        "minimizing the KL divergence KL(q | p).",
    )
    parser.add_argument(
        "--sample_kl",
        action="store_true",
        help="Whether to sample the KL divergence (otherwise: analytically).",
    )
    parser.add_argument(
        "--diagonal_prior",
        action="store_true",
        help="Whether to use a diagonal prior.",
    )
    parser.add_argument(
        "--predict_from_prior",
        action="store_true",
        help="Whether to additionally predict from the prior for PMF.",
    )
    parser.add_argument(
        "--number_of_samples_for_expectation",
        type=int,
        default=1,
        help="The number samples for the MC estimate.",
        metavar="K",
    )
    parser.add_argument(
        "--data_likelihood_std",
        "--data_likelihood_scale",
        type=float,
        default=0.15,
        help="Parameter for the likelihood of the model.",
    )
    parser.add_argument(
        "--batch_size", type=int, required=True, help="The batch size to use."
    )
    parser.add_argument(
        "--number_epochs",
        type=int,
        default=15000,
        help="Number of epochs to train the model for.",
    )
    parser.add_argument(
        "--validate_every_epochs",
        type=int,
        default=10,
        help="Interval of validating the model.",
    )
    parser.add_argument(
        "--print_loss_every_epochs",
        type=int,
        default=10,
        help="Interval of validating the model.",
    )
    parser.add_argument(
        "--checkpoint_every_epochs",
        type=int,
        default=-1,
        help="Interval of validating the model.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning-rate used with the optimizer.",
    )
    parser.add_argument(
        "--use_lr_schedule",
        action="store_true",
        help="Whether to use a learning-rate schedule.",
    )
    parser.add_argument(
        "--lr_schedule",
        choices=["RM", "Step", "Cyclical"],
        help="Scheduler to use for the learning-rate during training training.",
    )
    parser.add_argument(
        "--lr_schedule_a",
        type=float,
        default=1.0,
        help="Parameter for the learning-rate schedule.",
    )
    parser.add_argument(
        "--lr_schedule_b",
        type=float,
        default=1.0,
        help="Parameter for the learning-rate schedule.",
    )
    parser.add_argument(
        "--lr_schedule_rm_gamma",
        type=float,
        default=0.7,
        help="Parameter for the learning-rate schedule.",
    )
    parser.add_argument(
        "--lr_schedule_step_gamma",
        type=float,
        default=0.7,
        help="Parameter for the learning-rate schedule.",
    )
    parser.add_argument(
        "--lr_schedule_step_size",
        type=int,
        default=1000,
        help="Parameter for the learning-rate schedule.",
    )
    parser.add_argument(
        "--lr_schedule_warmup_fraction",
        type=float,
        default=0.25,
        help="Fraction of training epochs before scheduling begins.",
    )
    parser.add_argument(
        "--lr_scheduler_number_of_cycles",
        type=int,
        default=1,
        help="Number of cycles for cyclical learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_scheduler_min_lr",
        type=float,
        default=0.001,
        help="Minimal learning rate for Cyclical learnin rate scheduler.",
    )
    parser.add_argument(
        "--scheduler",
        choices=["Cyclical", "CyclicalConstant"],
        help="Scheduler to use.",
    )
    parser.add_argument(
        "--scheduler_number_of_cycles",
        type=int,
        default=1,
        help="Number of cycles for cyclic scheduler "
        "(1 equals no cycle, 2 linear function).",
    )
    parser.add_argument(
        "--scheduler_proportion_of_cycle_to_increase",
        type=float,
        default=0.5,
        help="Proportion of cycle to use for increasing the weight.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed (used for dataset-splits if --use_seed).",
    )
    parser.add_argument(
        "--use_seed",
        action="store_true",
        help="Whether to use a seed.",
    )
    parser.add_argument(
        "--graph_model",
        type=str,
        choices=graph_choices,
        help="Graph model to use in combination with MF.",
    )
    parser.add_argument(
        "--graph_featurizer",
        type=str,
        help="What kind of featurizer to use.",
        choices=graph_featurizer_choices,
    )
    parser.add_argument(
        "--graph_featurizer_atom_embedding_dim",
        type=int,
        help="Dimension of the embedding constructed by the featurizer for atoms.",
    )
    parser.add_argument(
        "--graph_featurizer_bond_embedding_dim",
        type=int,
        help="Dimension of the embedding constructed by the featurizer for bonds.",
    )
    parser.add_argument(
        "--graph_pmf_lin",
        action="store_true",
        help="Add a linear layer postprocessing the outputs (prior parameters) "
        "of the GNNs before the PMF.",
    )
    parser.add_argument(
        "--graph_pmf_lin_single_layer",
        action="store_true",
        help="Whether to use a single-layered NN postprocessing the outputs "
        "of the GNNs before the PMF.",
    )
    parser.add_argument(
        "--graph_lin_single_layer",
        action="store_true",
        help="Whether a single layer is used in the GNN (where applicable, e.g. FiLM)",
    )
    parser.add_argument(
        "--graph_in_channels",
        type=int,
        default=11,
        help="Dimensionality of one input node.",
    )
    parser.add_argument(
        "--graph_out_channels",
        type=int,
        default=1,
        help="Dimensionality of one output node.",
    )
    parser.add_argument(
        "--graph_hidden_channels",
        type=int,
        default=16,
        help="Dimensionality of one hidden node.",
    )
    parser.add_argument(
        "--graph_number_of_layers",
        type=int,
        default=3,
        help="Number of layers for the GNN.",
    )
    parser.add_argument(
        "--graph_dropout",
        type=float,
        default=0.0,
        help="Probability of dropout after each conv-layer.",
    )
    parser.add_argument(
        "--graph_activation",
        type=str,
        default="LeakyReLU",
        help="Activation function used in the GNN.",
        choices=graph_activation_choices,
    )
    parser.add_argument(
        "--graph_norm",
        type=str,
        default="none",
        help="Norm used in the GNN.",
        choices=graph_norm_choices,
    )
    parser.add_argument(
        "--leaky_relu_negative_slope",
        type=float,
        default=0.01,
        help="Negative slope of LeakyReLU.",
    )
    parser.add_argument(
        "--graph_final_prediction_agg",
        type=str,
        default="mean",
        help="How the final prediction is received from node features.",
        choices=["mean"],
    )
    parser.add_argument(
        "--graph_message_agg",
        type=str,
        default="mean",
        help="How the messages combined",
        choices=["add", "mean"],
    )
    parser.add_argument(
        "--graph_bias",
        action="store_true",
        help="Whether the GNN uses any biases.",
    )
    parser.add_argument(
        "--graph_initialize_bias_to_zero",
        action="store_true",
        help="Whether to initialize biases to 0 "
        "(applies to SumFormula and UNIFAC-Group-Contribution).",
    )
    parser.add_argument(
        "--graph_residual_every_layer",
        type=int,
        choices=[-1, 1, 2],
        help="Ue a residual connection every _ layer.",
    )
    parser.add_argument(
        "--graph_jumping_knowledge",
        type=str,
        default="last",
        help="The mode of JumpingKnowledge to use.",
    )
    parser.add_argument(
        "--graph_num_relations",
        type=int,
        default=16,
        help="Number of discrete edge labels (when applicable).",
    )

    arguments = parser.parse_args()
    return arguments
