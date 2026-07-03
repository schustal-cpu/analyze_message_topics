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
│   ├── data_utils.py
│   ├── preprocessing.py
│   ├── model_utils.py
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
| `analyze_data.ipynb`      | Executes the full analysis pipeline                                         |
| `src/data_utils.py`       | Functions for importing cleaning neccessary data                             |
| `src/preprocessing.py`    | Contains text preprocessing logic and an interactive stopword selection loop |
| `src/model_utils.py`      | Trains LDA and LSA models.              |
| `src/visualization.py`    | Responsible for visual outputs.                                               |

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
Download Yelp Dataset (Powershell)
```Poweshell
Invoke-WebRequest -Uri https://business.yelp.com/external-assets/files/Yelp-JSON.zip .\Yelp-JSON.zip
```
Extract Dataset and copy relevant file to project (cmd)
```cmd
tar -xf .\Yelp-JSON.zip
tar -xf .\Yelp-JSON\Yelp-JSON\yelp_dataset.tar
cp .\Yelp-JSON\Yelp-JSON\yelp_dataset\yelp_academic_dataset_review.json analyze_message_topics/data/
```
Run the pipeline in the following order:
1. Generate dataset
```bash
.\python get_data.py
```
2. Run analyze_data.ipynb in Jupyter Notebook

--- 
## Output Interpretation
The analysis produces:

- Extracted topics (grouped by sentiment)
- Top words per topic
- Comparison of LDA and LSA results

TODO: add example output (topics, keywords, scores, or plots)


## Notes and Limitations

Dataset size is limited to 1000 samples per source
Raw datasets are excluded due to data protection reasons

### Data Privacy
The following files are excluded via .gitignore:

- Raw datasets
- Generated CSV data
- External sentiment resources (GerVADER)


### Requirements

Python 3.14.3
Virtual environment (venv)

See requirements.txt for full dependencies.

---

## Data Usage and Licensing

This repository contains source code and references external datasets. Different components are subject to different licenses and usage terms.

### Source Code

The source code in this repository is licensed under the MIT License

This license applies **only to the code** and does not cover any external datasets.


### Yelp Open Dataset

This project uses the Yelp Open Dataset for academic, non-commercial purposes.

- The dataset is **not included** in this repository due to licensing restrictions.
- It must be downloaded manually from:
  https://business.yelp.com/data/resources/open-dataset/

Use of the dataset is subject to the Yelp Dataset Terms of Use:
https://www.yelp.com/dataset/terms

Key restrictions include:
- The data may only be used for **non-commercial purposes**
- The data may **not be redistributed or publicly shared**
- All rights to the dataset remain with Yelp
- User-generated content (e.g. reviews) must not be publicly exposed


### FragDenStaat Data

This project uses data from the FragDenStaat platform.  The usage of this data is subject to the FragDenStaat terms of use:
https://fragdenstaat.de/nutzungsbedingungen/

Key considerations:
- Data originates from public information requests but may still be subject to **usage and redistribution limitations**
- Users are responsible for ensuring compliance with applicable **data protection laws** and terms of use
- Personal or sensitive information must be handled appropriately


### Data Availability

Due to the above restrictions:

- No raw datasets are included in this repository
- Users must download and prepare the data themselves
- Generated datasets (e.g. CSV files) are excluded via `.gitignore`

---

### Disclaimer

This project is intended for educational and research purposes only.

The author does not claim any ownership of the external datasets.  
All rights remain with the respective providers (Yelp Inc. and FragDenStaat).

Users of this repository are responsible for ensuring that their usage complies with all applicable licenses and legal requirements.
