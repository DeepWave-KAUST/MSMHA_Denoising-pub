![LOGO](https://github.com/DeepWave-Kaust/MSMHA-dev/blob/main/asset/MSMHA_Architecture.png)

Reproducible material for **Unsupervised Multi-Stage Deep Learning Network for Seismic Data Denoising -Omar M. Saad, Matteo Ravasi and Tariq Alkhalifah**

[Click here](https://kaust.sharepoint.com/:f:/r/sites/M365_Deepwave_Documents/Shared%20Documents/Restricted%20Area/REPORTS/DW0046?csf=1&web=1&e=natLJ3) to access the Project Report. Authentication to the _Restricted Area_ filespace is required.

# Project structure
This repository is organized as follows:

* :open_file_folder: **asset**: folder containing logo;
* :open_file_folder: **data**: folder containing data (or instructions on how to retrieve the data

## Notebooks
The following notebooks are provided:

- :orange_book: ``MSMHA.ipynb``: notebook performing the denoising;
- :orange_book: ``Utils.ipynb``: notebook including models, patching, and unpatching scripts;



## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `MSMHA.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. 

Remember to always activate the environment by typing:
```
conda activate MSMHA
```

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce RTX 3090 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.

## Cite us 
DW0046 - O. M. Saad, M. Ravasi, T. Alkhalifah (2024) Unsupervised Multi-Stage Deep Learning Network for Seismic Data Denoising.

