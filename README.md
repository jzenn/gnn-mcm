# [Balancing Molecular Information and Empirical Data in the Prediction of Physico-Chemical Properties](http://arxiv.org/abs/2406.08075)
<div id="top"></div>

  [![arxiv-link](https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red)](http://arxiv.org/abs/2406.08075)
  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14414242.svg)](https://doi.org/10.5281/zenodo.14414242)
  [![made-with-pytorch](https://img.shields.io/badge/Made%20with-PyTorch-brightgreen)](https://pytorch.org/)
  [![made-with-pytorch](https://img.shields.io/badge/Made%20with-PyTorch%A0Geometric-brightgreen)](https://pytorch.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

  <a href="https://jzenn.github.io" target="_blank">Johannes&nbsp;Zenn</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://mv.rptu.de/fgs/ltd/lehrstuhl/mitarbeiter/dominik-gond" target="_blank">Dominik&nbsp;Gond</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://mv.rptu.de/en/dpts/ltd/chair/staff/fabian-jirasek" target="_blank">Fabian&nbsp;Jirasek</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://robamler.github.io" target="_blank">Robert&nbsp;Bamler</a>



## About The Project
This is the official GitHub repository for our work [Balancing Molecular Information 
and Empirical Data in the Prediction of Physico-Chemical 
Properties](http://arxiv.org/abs/2406.08075) 
where we propose a hybrid method for combining molecular descriptors with 
representation learning for the (exemplary) task of predicting activity coefficients.

> Predicting the physico-chemical properties of pure substances and mixtures is a 
> central task in thermodynamics. Established prediction methods range from fully 
> physics-based ab-initio calculations, which are only feasible for very simple 
> systems, over descriptor-based methods that use some information on the molecules 
> to be modeled together with fitted model parameters (e.g., 
> quantitative-structure-property relationship methods or classical group 
> contribution methods), to representation-learning methods, which may, in extreme 
> cases, completely ignore molecular descriptors and extrapolate only from existing 
> data on the property to be modeled (e.g., matrix completion methods). In this work, 
> we propose a general method for combining molecular descriptors with representation 
> learning using the so-called expectation maximization algorithm from the 
> probabilistic machine learning literature, which uses uncertainty estimates to 
> trade off between the two approaches. The proposed hybrid model exploits chemical 
> structure information using graph neural networks, but it automatically detects 
> cases where structure-based predictions are unreliable, in which case it corrects
> them by representation-learning based predictions that can better specialize to 
> unusual cases. The effectiveness of the proposed method is demonstrated using the 
> prediction of activity coefficients in binary mixtures as an example. The results 
> are compelling, as the method significantly improves predictive accuracy over the 
> current state of the art, showcasing its potential to advance the prediction of 
> physico-chemical properties in general.


## Step 1 of 4: Installation (Virtual Environment)

We recommend using a virtual environment to avoid dealing with other packages 
installed in the system.
First, clone the repository using `git clone git@github.com:jzenn/gnn-mcm.git` 
and navigate to the repository folder with `cd gnn-mcm`.
You can install a virtual environment via either of the two methods given below.


### Environment: `conda` (recommended)
 
- install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
- create a new environment `conda create python=3.9 --name gnn-mcm`
- activate the environment `conda activate gnn-mcm`


### Environment: `venv`

Please make sure that Python 3.9 is used for the installation, otherwise one might 
run into version conflicts with `torch`.

- create a new environment `python3.9 -m venv venv`
- activate the environment `source venv/bin/activate`


## Step 2 of 4: Installation (Requirements and Packages)

First, make sure that `pip==24.3.1` is installed (`pip install --upgrade pip==24.3.1`).
Then, install the `requirements.txt` via
```bash
pip install -r requirements.txt
```
After all requirements have been installed, run the following.
```bash
# replace torch-1.10.0+cpu by torch-1.10.0+{cu102,cu113,cu111}
# depending on availability of accelerator
pip install torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.10.0+cpu.html
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.10.0+cpu.html
pip install torch-sparse==0.6.13 -f https://data.pyg.org/whl/torch-1.10.0+cpu.html
pip install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.10.0+cpu.html
```


## Step 3 of 4: Preparing the Data

We provide a detailed description of the data preparation process in the following.


### Data Used in Medina et al. (2022)

We provide the processed `CSV` file for the data used in [Medina et al. (2022)](
https://pubs.rsc.org/en/content/articlehtml/2022/dd/d1dd00037c) (taken from [their 
repository](https://github.com/edgarsmdn/GNN_IAC)) at 
`data/medina_2022/medina_data.csv`.
Additionally, we provide the `JSON` file that contains the embeddings for the 
molecules.


### Dortmund Data Bank 2019 (optional)

The Dortmund Data Bank 2019 (DDB) is not publicly available but can be downloaded 
with a paid subscription.
If you want to train on the DDB dataset, the data should be processed as described in 
[Jirasek et al. (2020)](https://pubs.acs.org/doi/full/10.1021/acs.jpclett.9b03657?casa_token=rjHtKSC14XwAAAAA%3A6bjf1zVHTfwKy9_pfLSx6kigu-hl3-5rMHTlqelM-9QBw5Dn_ZIuyN-G0vt_Q5daYuRYt42Tx5T0P7z-). 
The provided `CSV` file should contain the following columns:
- `log_gamma_exp`: log of the experimental activity coefficient
- `solute_idx`: index of the solute
- `solvent_idx`: index of the solvent
- `solute_smiles`: SMILES string of the solute
- `solvent_smiles`: SMILES string of the solvent

The values of the `*_smiles` keys are matched with the corresponding objects in a 
`JSON` file.
The `JSON` file has the same structure as for the data of Medina et al. (2022) 
(see `data/medina_2022/medina_data.csv` and 
`data/medina_2022/featurized_molecules.json`).


## Step 4 of 4: Running the Code (Exemplary Script)

The script `train_medina_example.sh` provides an executable script for training the
GNN-MCM on the dataset used by Medina et al. (2022).
You can use this script by running
```bash
./train_medina_example.sh
```
to test whether the
installed libraries work, but the resulting trained model will not be useful because
real training requires a lot more training epochs.
If this script runs for a few minutes and then prints test results (including test 
MSE and MAE) to the terminal, then your installation works.

To replicate the results in our paper, run a command of the following form,
```bash
python main.py <arguments for training>
```
where the exact `<arguments for training>` that we used in our experiments are listed
in the files in the directory `hyperparameters`.
When taking these arguments, make sure that you
- replace each `<insert-path>` by a suitable path (cf. example in file 
`train_medina_example.sh`);
- replace `<insert-name>` with an identifier of your choice (the training script will 
create a subdirectory with this name in the directory specified by
`--experiment_base_path`, where it will store checkpoints and results);
- replace `<ensemble-id>` with a number from 1 to 10 to specify the current 
train/test split for 10-fold cross validation;
- replace `<M>` and `<N>` by the number of solutes and solvents that the dataset 
contains;
- replace `<M'>` and `<N'>` (if present) by the number of solutes and solvents that 
should be excluded from the training set for zero-shot prediction;
- concatenate all arguments into a single space-separated line.


## License
Distributed under the MIT License. See `LICENSE.MIT` for more information.


## Citation

```bibtex
@article{zenn2024balancing,
  title={Balancing Molecular Information and Empirical Data in the Prediction of Physico-Chemical Properties}, 
  author={Johannes Zenn and Dominik Gond and Fabian Jirasek and Robert Bamler},
  journal={arXiv preprint arXiv:2406.08075},
  year={2024}
}
```

<p align="right">(<a href="#top">back to top</a>)</p>
