# Mechanistic Model for Organic Reactions
Machine learning models for mechanism prediction of organic reaction with WLDN, Transformer, and Graph2SMILES, each residing in its respective folder.
This project is based on the paper: https://doi.org/10.1002/anie.202411296.

## Choose the model
There are three machine learning models designed for the task of predicting organic reaction mechanisms: 
- WLDN (Weisfeiler-Lehman Difference Network): Located in the WLDN folder.
- Transformer: Located in the Transformer folder.
- Graph2SMILES: Located in the Graph2SMILES folder.

Each model folder contains a dedicated README.md file that provides detailed instructions on how to set up and use the model. Please navigate to the model folder of your choice and follow the instructions provided in the README.md.

## Data preparation
The mechanistic dataset generated from the USPTO-Full dataset is available for download. This dataset includes mechanistic pathways for a wide range of chemical reactions.
You can download the dataset from Figshare: https://doi.org/10.6084/m9.figshare.26046106.v1. 

Alternatively, you can create your own mechanistic dataset using the script in `Mechanistic_dataset_preparation/Mechanistic_dataset_generation.ipynb`.
