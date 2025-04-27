# Amazon seviews sentiment analysis
This project focuses on analyzing and classifying sentiment in a large dataset of over 100,000 Amazon product reviews. By conducting exploratory data analysis (EDA) and applying advanced Natural Language Processing (NLP) techniques, the goal is to build a sentiment classification model capable of accurately identifying customer sentiment as either positive or negative. Several models were developed and evaluated to find the most efficient approach for this task.

## Features
- Exploratory Data Analysis (EDA): Performed thorough analysis on 100,000+ Amazon reviews to understand the dataset's structure and key patterns.

- Text Preprocessing: Applied essential NLP techniques, including tokenization, lemmatization, and removal of stopwords, to clean and prepare the review text.

- Model Development: Developed multiple machine learning models for sentiment classification, including:
    * Logistic Regression

    * Support Vector Classification (SVC)

    * BERT (Bidirectional Encoder Representations from Transformers) for comparison.

- Model Evaluation: Compared the performance of traditional models like Logistic Regression and SVC with BERT to determine the best approach for the given task.

## Technologies Used
- Python: Programming language for data processing, model building, and evaluation.

- Pandas: Data manipulation and analysis.

- NumPy: Numerical computation.

- Scikit-learn: Machine learning models (Logistic Regression, SVC) and evaluation metrics.

- NLTK : Natural language processing techniques such as tokenization, stopword removal, and lemmatization.

- TfidfVectorizer: For converting text data into TF-IDF features.

- Transformers (Hugging Face): For implementing BERT-based models.

- Matplotlib / Seaborn: For data visualization and model performance plots.