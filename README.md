# Paper ID: 8608 - ST-Prompt: Group-wise Prompting for Synthetic Tabular Data Generation using Large Language Models

## Introduction

This is the implementation of "ST-Prompt: Group-wise Prompting for Synthetic Tabular Data Generation using Large Language Models."


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
     -  We provide preprocessed data in `data/realdata/Sick`, which downloaded from the [download link](https://www.openml.org/search?type=data&sort=runs&id=38&status=active). 
     
### Usage

1. (Optional) **Configure OpenAI API Key**: Enter your OpenAI API key in `codes/SyntheticDataGeneration/generate_samples_Sick.py`:

   ```python
   (line 13) openai_key = "Your-OpenAI-Key"
   ```

2. (Optional) **Generate Synthetic Datasets with ST-Prompt**: 

   To generate synthetic datasets using ST-Prompt, run the following command:

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
