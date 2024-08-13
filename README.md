# Tutorbot
## Overview

A chatbot based on Retrieval-Augmented Generation architecture and pre-trained LLM models. Different combinations of Language models (BERT variants, Phi variants, etc.) and retriever embedding models are evaluated to get the optimal RAG pipeline. This project aims to address the challenge of limited interaction in online learning. 

## Getting Started

### Prerequisites

1. Clone the repository:
```bash
git clone <clone link>
```
2. Install necessary Python package dependencies before running the notebooks:
```bash
pip install -r requirements.txt
```
3. Python 3.10 is a prerequisite to run the code

4. Few LLM Models require CUDA GPU environments to run. 16GB GPU environment is recommended.
   
5. Please provide project root paths when asked in the Jupyter notebooks 

### Project Structure

- `/Dataset`: Contains data sets.
- `/LLM`:     Contains different language models (experimented on train set)
- `/Information_Retrievers`: Contains different Retrievers models (experimented on train set)
- DataPreProcess.ipynb will populate train, test sets & raw knowledge articles under /Dataset.
- GenerateAnswers.ipynb will generate synthetic ground truth answers for train/test sets.
- RAGEval.ipynb to evaluate RAG pipelines End-End using test set
- `/demo_interface`: Contains streamlit based chatbot (which can run on local host) to answer student questions. Additionally, the chatbot is hosted on `http://34.69.184.232:8000/`

### High level Code walkthrough

- Optionally Rerun `DataPreProcess.ipynb` (since dataset is already present) to populate datasets and knowledge base under `/Dataset`.
- Optionally Rerun `GenerateAnswers.ipynb` (since already generated) to generate synthetic ground truth answers.
- Run notebooks under `/LLM` to experiment prompt engineering for different generator models.
- Run notebooks under `/Information_Retrievers` to experiment and benchmark retrievers to get best retriever model.
- Finally run RAGEval.ipynb to benchmark RAG pipelines using generator models (with best retriever) using test dataset.

### Notes

- LLM prompts are created with the aid of ChatGPT
- Raw knowledge articles used in this project to extract the dataset are from https://en.wikipedia.org/wiki/Data_science which is open source.

### Contributors
1. Tung Nguyen: ngtung@umich.edu
2. Mnatsakan Sharafyan: sharaf@umich.edu
3. Sangram Sanjiv Malekar: sangram@umich.edu
