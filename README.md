# Tutorbot
## Overview

A chatbot based on Retrieval-Augmented Generation architecture and pre-trained BERT models. This project aims to address the challenge of limited interaction in online learning. 

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

### Project Structure

- `/Dataset`: Contains data sets.
- `/LLM`:     Contains different language models (experimented on train set)
- `/Information_Retrievers`: Contains different Retrievers models (experimented on train set)
- DataPreProcess.ipynb will populate train, test sets & raw knowledge articles under /Dataset.
- GenerateAnswers.ipynb will generate synthetic ground truth answers for train/test sets.
- RAGEval.ipynb to evaluate RAG pipelines End-End using test set
