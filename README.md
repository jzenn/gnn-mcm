# [Balancing Molecular Information and Empirical Data in the Prediction of Physico-Chemical Properties](http://arxiv.org/abs/2406.08075)
<div id="top"></div>

  [![arxiv-link](https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red)](http://arxiv.org/abs/2406.08075)
  [![DOI](https://zenodo.org/badge/814016378.svg)](https://zenodo.org/doi/10.5281/zenodo.11631566)
  [![made-with-pytorch](https://img.shields.io/badge/Made%20with-PyTorch-brightgreen)](https://pytorch.org/)
  [![made-with-pytorch](https://img.shields.io/badge/Made%20with-PyTorch%A0Geometric-brightgreen)](https://pytorch.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

  <a href="https://jzenn.github.io" target="_blank">Johannes&nbsp;Zenn</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://mv.rptu.de/fgs/ltd/lehrstuhl/mitarbeiter/dominik-gond" target="_blank">Dominik&nbsp;Gond</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://mv.rptu.de/en/dpts/ltd/chair/staff/fabian-jirasek" target="_blank">Fabian&nbsp;Jirasek</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://robamler.github.io" target="_blank">Robert&nbsp;Bamler</a>



## About The Project
This is the official GitHub repository for our work [Balancing Molecular Information and Empirical Data in the Prediction of Physico-Chemical Properties](http://arxiv.org/abs/2406.08075) where we propose a hybrid method for combining molecular descriptors with representation learning for the (exemplary) task of predicting activity coefficients.

> Predicting the physico-chemical properties of pure substances and mixtures is a central task in thermodynamics. Established prediction methods range from fully physics-based ab-initio calculations, which are only feasible for very simple systems, over descriptor-based methods that use some information on the molecules to be modeled together with fitted model parameters (e.g., quantitative-structure-property relationship methods or classical group contribution methods), to representation-learning methods, which may, in extreme cases, completely ignore molecular descriptors and extrapolate only from existing data on the property to be modeled (e.g., matrix completion methods). In this work, we propose a general method for combining molecular descriptors with representation learning using the so-called expectation maximization algorithm from the probabilistic machine learning literature, which uses uncertainty estimates to trade off between the two approaches. The proposed hybrid model exploits chemical structure information using graph neural networks, but it automatically detects cases where structure-based predictions are unreliable, in which case it corrects them by representation-learning based predictions that can better specialize to unusual cases. The effectiveness of the proposed method is demonstrated using the prediction of activity coefficients in binary mixtures as an example. The results are compelling, as the method significantly improves predictive accuracy over the current state of the art, showcasing its potential to advance the prediction of physico-chemical properties in general.
> 


## Installation (Virtual Environment)

We recommend using a virtual environment to avoid dealing with other packages installed in the system.
First, clone the repository using `git clone git@github.com:jzenn/gnn-mcm.git` and navigate to the repository folder with `cd gnn-mcm`.
Two choices for installing virtual environments are given below.



### Environment: `conda` (recommended)
 
- install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
- create a new environment `conda create python=3.9 --name gnn-mcm`
- activate the environment `conda activate gnn-mcm`


### Environment: `venv`

Please make sure that Python 3.9 is used for the installation, otherwise one might run into version conflicts with `torch`.

- create a new environment `python3.9 -m venv gnn-mcm`
- activate the environment `source gnn-mcm/bin/activate`


## Installation (Requirenents and Packages)

The dependencies and packages can be installed using the provided `requirements.txt` file using `pip install -r requirements.txt`.


## Preparing the Data

We provide a detailed description of the data preparation process in the following.


### Dortmund Data Bank 2019

The Dortmund Data Bank 2019 (DDB) is not publicly available but can be downloaded with 
a paid subscription.
The dataset is processed as described in Jirasek et al. (2020). 
The provided `CSV` file should contain the following columns:
- `log_gamma_exp`: log of the experimental activity coefficient
- `solute_idx`: index of the solute
- `solvent_idx`: index of the solvent
- `solute_smiles`: SMILES string of the solute
- `solvent_smiles`: SMILES string of the solvent

The values of the `*_smiles` keys are matched with the corresponding objects in a `JSON` file.
The `JSON` file has the same structure as for the data of Medina et al. (2022) (see below).


### Data Used in Medina et al. (2022)

We provide the processed `CSV` file for the data used in Medina et al. (2022) in the `data` folder.
Additionally, we provide the `JSON` file that contains data and embeddings for the molecules.


## Running the Code

Example training command for DAIS baseline:
```bash
python main.py <arguments for training>
```
`<arguments for training>` can be replaced by the files found in the `hyperparameters` folder.


## License
Distributed under the MIT License. See `LICENSE.MIT` for more information.


## Citation:
Following is the Bibtex if you would like to cite our paper :

```bibtex
@article{zenn2024balancing,
  title={Balancing Molecular Information and Empirical Data in the Prediction of Physico-Chemical Properties}, 
  author={Johannes Zenn and Dominik Gond and Fabian Jirasek and Robert Bamler},
  journal={arXiv preprint arXiv:2406.08075},
  year={2024}
}
```

<p align="right">(<a href="#top">back to top</a>)</p>
