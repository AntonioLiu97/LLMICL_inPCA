## LLMICL InPCA

![sigma_0.1_0.3_0.5_0.7_traj_Hellinger_2D](./figures/randomPDF_70B_snapshots.gif)

## Overview
This repository contains the complementary code base for the paper: 
    [Density estimation with LLMs: a geometric investigation of in-context learning trajectories](https://arxiv.org/abs/2410.05218)

## Directory structure

- `/data`: Contains functions for converting lists of sampled data $X_1,X_2,...,X_n \sim P(x)$ data into 1D strings, which is then used to prompt LLMs.
It contains `series_generator.ipynb`, a Jupyter notebook for generating all distributions investigated in the paper: Gaussian, uniform, Student's t-distribution, random PDFs.

- `/generated_series`: This directory caches all prompts generated by the `series_generator.ipynb`, in the form of pickled dictionaries.

- `/models`: 
    - `ICL.py` implements essential packages like Hierarchy-PDF and its auxiliary functions. 
    - `generate_predictions.py` prompts the LLM, such as LLaMA, Mistral, and Gemma with the generated prompts and save the estimated PDF as pickled Hierarchy-PDF.
    - `baseline_models.py` implements baseline density-estimation algorithm such as KDE and bayesian histogram

- `/processed_series`: Stores the density estimation trajectories of LLMs

- `/inPCA`: Contains jupyter notebooks for analyzing LLMs' DE trajectories with InPCA
    - `inPCA_multi_traj.ipynb` simultaneously embeds multiple DE trajectories within the same inPCA visualization
    - `inPCA_multi_traj_kernel_nD_fit.ipynb` simultaneously embeds multiple DE trajectories, as well as their bespoke KDE trajectories.
    - `inPCA_multi_traj_kernel_nD_fit_meta_embed.ipynb` 
    performs meta-inPCA embeddings of multiple trajectories and their bespoke KDE imitations.

- `/figures`: A repository for all figures generated through the analysis processes.




