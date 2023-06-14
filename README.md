# Table of Contents

* [Introduction](#introduction)
* [SCANNet Model](#DeepAt-framework)
* [Usage](#usage)
* [Datasets](#datasets)
* [References](#references)

<a name="introduction"></a>

# Introduction
This repository is the official implementation of [Deep learning reveals key aspects to help interpret the structure–property
relationships of materials](https://).

Please cite us as

```

```

We developed a `Self-Consistent Atention-based Neural Network (SCANNet)` that takes advantage of a neural network to quantitatively capture
the contribution of the local structure of material properties.

The model captures information on atomic sites
and their local environments by considering self-consistent long-range interactions to enrich the structural
representations of the materials. A comparative experiment was performed on benchmark dataset QM9 to compare
the performance of the proposed model with state-of-the-art representations in terms of prediction accuracy
for several target properties.

Furthermore,
the quantitative contribution of each local structure to the properties of the materials can help understand
the structural-property relationships of the materials.

<a name="DeepAt-framework"></a>

# SCANNet framework

The Self-Consistent Atention-based Neural Network (SCANNet) is an implementation of deep attention mechanism for materials science

Figure 1 shows the overall schematic of the model

![Model architecture](resources/model_semantic.jpg)
<div align='center'><strong>Figure 1. Schematic of  SCANNet.</strong></div>

<a name="usage"></a>

# Usage

Our current implementation supports a variety of use cases for users with
different requirements and experience with deep learning. Please also visit
the [notebooks directory](notebooks) for Jupyter notebooks with more detailed code examples.

## Using pre-built models

In our work, we have already built models for the QM9 data set [1]. The model is provided as serialized HDF5+JSON files. 

* QM9 molecule data:
  * HOMO: Highest occupied molecular orbital energy
  * LUMO: Lowest unoccupied molecular orbital energy
  * Gap: energy gap
  * α: isotropic polarizability
  * Cv: heat capacity at 298 K

The MAEs on the various models are given below:

### Performance of QM9 MEGNet-Simple models

| Property | Units      | MAE   |
|----------|------------|-------|
| HOMO     | meV         | 41 |
| LUMO     | meV         | 37 |
| Gap      | meV         | 61 |
| α        | Bohr^3     | 0.141|
| Cv       | cal/(molK) | 0.050 |

<a name="dataset"></a>

# Datasets

## Experiments

The settings for experiments specific is placed in the folder [configs](configs)

We provide an implementation for the QM9 experiments, the fullerene-MD, the Pt/graphene-MD and SmFe12-MD experiments

# Basic usage
## Model training
For training new model for QM9 dataset, please follow the below example scripts. If the data for QM9 is not avaiable, please run the code ```preprocess_data.py``` for downloading and creating suitable data formats for SCANNet model.
```
python preprocess_data.py qm9 processed_data --dt=4.0 --wt=0.2
```
The data for QM9 will be processed and saved into folder [propessed_data](processed_data).
After that, please change the config file located in folder [configs](configs) for customizing the model hyperparameters or data loading/saving path.
```
python train.py homo configs/model_qm9.yaml --use_ring=True
```

For training dataset fullerene-MD with pretrained weights from QM9 dataset, please follow these steps. The pretrained model will be load based on the path from argument. 
```
python preprocess_data.py fullerene processed_data --data_path='experiments/fullerene' --dt=4.0 --wt=0.2
...
python train.py homo configs/model_fullerene.yaml --use_ring=True --pretrained=..../qm9/homo/models/model.h5
```
## Model inference
The code ```predict_files.py``` supports loading a ```xyz``` file and predicting the properties with the pretrained models. The information about global attention (GA) score for interpreting the structure-property relationship is also provided and saved into ```xyz``` format. Please use a visualization tool such as Ovito [2] for showing the results.
```
python predict_files.py ..../models.h5 save_path.../ experiments/molecules/Dimethyl_fumarate.xyz
``` 
![Visualization of GA scores](resources/ovito_visual.png)
<div align='center'><strong>Figure 2. Example of SCANNet prediction for LUMO property.</strong></div>

<a name="usage"></a>

<a name="references"></a>
# References

[1] Ramakrishnan, R., Dral, P., Rupp, M. et al. Quantum chemistry structures and properties of 134 kilo molecules. Sci Data 1, 140022 (2014). https://doi.org/10.1038/sdata.2014.22 

[2] A. Stukowski, Visualization and Analysis of Atomistic Simulation Data with OVITO–the Open Visualization Tool, Model. Simul. Mater. Sci. Eng. 18, 15012 (2009). [doi:10.1088/0965-0393/18/1/015012](https://stacks.iop.org/0965-0393/18/015012)