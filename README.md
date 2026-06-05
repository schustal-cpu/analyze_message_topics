# analyze_message_topics

A project for extracting and analyzing dominant topics and their sentiment (positive/negative) from unstructured text data.  
The pipeline supports both German and English text and works across formal complaints and social media messages.

---

## Motivation

The goal of this project is to identify the most relevant topics in textual data and assess whether they are discussed positively or negatively.

The system is designed to:
- work with different types of input (formal requests and social media)
- support multilingual text (German and English)
- enable structured analysis of unstructured data
- test if LDA or LSA works best for the selected data

---

## Features

- Topic extraction using LDA and LSA
- Sentiment analysis using VADER and GerVADER
- Multilingual support (German and English)
- Preprocessing with tokenization, stopword removal, and lemmatization
- Feature extraction using Bag-of-Words and TF-IDF
- Comparison of different topic modeling approaches

---

## Project Structure
```
analyze_message_topics/
│── data/                  # Generated and raw datasets (excluded via .gitignore)
│── src/                   # Core processing modules
│   ├── analyzer.py
│   ├── preprocessing.py
│   ├── training.py
│   ├── visualization.py
│   └── styles.py
│── get_data.py            # Data collection & preparation
│── analyze_data.py        # Main analysis pipeline
│── requirements.txt
│── README.md
```

---

## Data Sources

The project uses two different datasets:

- Formal complaints (German)  
  Source: https://fragdenstaat.de/api/v1/request/  
  Automatically collected using get_data.py

- Social media messages (English)  
  Source: https://business.yelp.com/data/resources/open-dataset/  
  Must be downloaded manually and stored locally

---

## Data Preparation

The script get_data.py performs the following steps:

- Downloads or loads raw data
- Extracts relevant fields
- Limits dataset to 1000 entries per source
- Creates a unified CSV file with the structure:


origin, id, created, review
Yelp/FragDenStaat, int, timestamp, text

Output is stored in:
data/combined_reviews.csv

---

## Processing Pipeline

The analysis pipeline (analyze_data.py) includes the following steps:

### 1. Text Preprocessing
- Tokenization (NLTK)
- Stopword removal (including custom stopwords)
- Lemmatization
- Lowercasing
- Optional: repeated stopword filtering and n-grams

### 2. Sentiment Analysis
- VADER (English)
- GerVADER (German)
- Classification into positive and negative texts

### 3. Vectorization
- Bag-of-Words
- TF-IDF (scikit-learn)

### 4. Topic Modeling
- Latent Semantic Analysis (LSA)
- Latent Dirichlet Allocation (LDA)

### 5. Evaluation
- Comparison of LSA vs LDA results  
- TODO: describe evaluation metric (e.g. coherence, manual review, etc.)

---

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
```
Activate environment:
```bash
.\venv\Scripts\activate
```
Install requirements:
```bash
pip install -r requirements.txt
```

Usage
Run the pipeline in the following order:
1. Generate dataset
```bash
.\python get_data.py
```
2. Run analysis
```bash
.\python analyze_data.py
```
Output
The analysis produces:

Extracted topics (grouped by sentiment)
Top words per topic
Comparison of LDA and LSA results

TODO: add example output (topics, keywords, scores, or plots)

## Module Overview

| Name                     | Description                                                                 |
|--------------------------|------------------------------------------------------------------------------|
| `get_data.py`             | Handles data collection and transformation into CSV format                  |
| `analyze_data.py`         | Executes the full analysis pipeline                                         |
| `src/preprocessing.py`    | Contains text preprocessing logic. TODO: list main functions                 |
| `src/analyzer.py`         | Handles sentiment classification and topic assignment. TODO: clarify responsibilities |
| `src/training.py`         | Trains LDA and LSA models. TODO: specify parameters / configs                |
| `src/visualization.py`    | Responsible for visual outputs. TODO: describe generated plots               |


## Notes and Limitations

Dataset size is limited to 1000 samples per source
Raw datasets are excluded due to data protection reasons

### Data Privacy
The following files are excluded via .gitignore:

- Raw datasets
- Generated CSV data
- External sentiment resources (GerVADER)


### Possible Improvements
TODO:

Improve evaluation strategy for topics
Add automated metrics (coherence score, perplexity)
Extend multilingual support
Integrate advanced models


### Requirements

Python 3.14.3
Virtual environment (venv)

See requirements.txt for full dependencies.

### License
TODO: Add license information
