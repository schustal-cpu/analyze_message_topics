# analyze_message_topics

A project for extracting and analyzing dominant topics and their sentiment (positive/negative) from unstructured text data.  
The pipeline supports both German and English text and works across formal complaints and social media messages.

---

## Features

- structured analysis of unstructured data with diffenrent types ( formal request and social media )
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
│── data/                  # Generated and raw datasets (excluded via .gitignore), stopwords
│── src/                   # Core processing modules
│   ├── analyzer.py
│   ├── preprocessing.py
│   ├── training.py
│   ├── visualization.py
│   └── styles.py
│── get_data.py            # Data collection & preparation
│── analyze_data.ipynb     # Main analysis pipeline
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

- Downloads or loads raw data from "fragdenstaat.de"
- Imports Yelp dataset from data/yelp_academic_dataset_review.json
- Extracts relevant fields
- Limits dataset to 1000 entries per source
- Creates a unified CSV file with the structure:

data/combined_reviews.csv	# exluded for dataprotection by .gitignore
| origin | id | created | review |
|--- | --- | --- | --- |
| Yelp/FragDenStaat | int | timestamp | text |

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

## Module Overview

| Name                     | Description                                                                 |
|--------------------------|------------------------------------------------------------------------------|
| `get_data.py`             | Handles data collection and transformation into CSV format                  |
| `analyze_data.py`         | Executes the full analysis pipeline                                         |
| `src/preprocessing.py`    | Contains text preprocessing logic. TODO: list main functions                 |
| `src/analyzer.py`         | Handles sentiment classification and topic assignment. TODO: clarify responsibilities |
| `src/training.py`         | Trains LDA and LSA models. TODO: specify parameters / configs                |
| `src/visualization.py`    | Responsible for visual outputs. TODO: describe generated plots               |

---

## Install

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

## Usage
Download Yelp Dataset
```Poweshell
(PS) Invoke-WebRequest -Uri https://business.yelp.com/external-assets/files/Yelp-JSON.zip .\Yelp-JSON.zip
```
```cmd
(CMD) tar -xf .\Yelp-JSON.zip
(CMD) tar -xf .\Yelp-JSON\Yelp-JSON\yelp_dataset.tar
(CMD) cp .\Yelp-JSON\Yelp-JSON\yelp_dataset\yelp_academic_dataset_review.json analyze_message_topics/data/
```

Run the pipeline in the following order:
1. Generate dataset
```bash
.\python get_data.py
```
2. Run analysis in Jupyter Notebook
```bash
.\python analyze_data.py
```
## Output
The analysis produces:

Extracted topics (grouped by sentiment)
Top words per topic
Comparison of LDA and LSA results

--- 

## Interpretation

TODO: add example output (topics, keywords, scores, or plots)


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
