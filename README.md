# Paper ID: 8608 - ST-Prompt: Group-wise Prompting for Synthetic Tabular Data Generation using Large Language Models

## Introduction

This is the implementation of "ST-Prompt: Group-wise Prompting for Synthetic Tabular Data Generation using Large Language Models."


## Environment Setup

This code was developed using Python 3.8 on an Ubuntu 18.04 system.

## Quick start

### Installation

1. **Install Required Packages**:
   Use `pip` to install the necessary Python packages from the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```


2. **Data Preparation**:
   
   - **Obtain the dataset**: 
     - Sick: 링크 
     - ... TODO
   
   - **Organize the dataset**: Move the downloaded dataset to the following directory structure:
     ```
     Sick: data/realdata/Sick/dataset_38_sick.arff
     ```
   
   - **Run preprocessing**: Navigate to the preprocessing code directory and execute the preprocessing script:
     ```bash
     cd codes/DataProcessing/
     python DataSplit_Sick.py
     다른데이터셋 TODO
     cd ..
     ```
   
### How to use

1. Your OpenAI API Key를 입력하세요.

2. **Generating synthetic dataset with ST-Prompt**:
   
   To generate synthetic datasets, execute the following command:
   ```
   TODO
   cd 
   ```
3. **Training and evaluating downstream task models**:

   TODO