# Group-wise Prompting for Synthetic Tabular Data Generation using Large Language Models

## Introduction

This is the implementation of "Group-wise Prompting for Synthetic Tabular Data Generation using Large Language Models."


## Environment Setup

This code was developed using Python 3.8 on Ubuntu 18.04.

## Quick start

### Installation

1. **Install Required Packages**:
   Use `pip` to install the necessary Python packages from the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```


2. **Data Preparation**:
- The Sick dataset is available for download at the following [download link](https://www.openml.org/search?type=data&sort=runs&id=38&status=active).  
- The preprocessed version of the Sick dataset can be found in the directory `data/realdata/Sick`.

### Usage

1. (Optional) **Configure OpenAI API Key**: Enter your OpenAI API key in `codes/SyntheticDataGeneration/generate_samples_Sick.py`:

   ```python
   (line 13) openai_key = "Your-OpenAI-Key"
   ```

2. (Optional) **Generate Synthetic Datasets**: 

   To generate synthetic datasets using our method, run the following command:

   ```bash
   cd SyntheticDataGeneration
   python generate_samples_Sick.py
   cd ..
   ```

3. **Train and Evaluate Downstream Task Models**:
   To evaluate the quality of the synthetic data, use the following command:

   ```bash
   cd DownstreamTasks
   python Classification.py    
   ```


## Citation

If you find this work or code is helpful in your research, please cite:
```
@article{kim2024group,
  title={Group-wise Prompting for Synthetic Tabular Data Generation using Large Language Models},
  author={Kim, Jinhee and Kim, Taesung and Choo, Jaegul},
  journal={arXiv preprint arXiv:2404.12404},
  year={2024}
}
```
