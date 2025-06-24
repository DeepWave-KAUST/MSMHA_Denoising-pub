![LOGO](https://github.com/DeepWave-Kaust/MSMHA-dev/blob/main/asset/MSMHA_Architecture.png)

Reproducible material for **Self-supervised multi-stage deep learning network for seismic data denoising -Omar M. Saad, Matteo Ravasi, and Tariq Alkhalifah**

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
```bibtex
@article{saad2025self,
  title={Self-supervised multi-stage deep learning network for seismic data denoising},
  author={Saad, Omar M and Ravasi, Matteo and Alkhalifah, Tariq},
  journal={Artificial Intelligence in Geosciences},
  pages={100123},
  year={2025},
  publisher={Elsevier}
}

