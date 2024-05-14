# Reddit Comments Classification

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Database Connection](#database-connection)
  - [Environment Variables](#environment-variables)
  - [Data Sampling](#data-sampling)
  - [Classification](#classification)
  - [Data Cleaning](#data-cleaning)
  - [Model Training](#model-training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project classifies Reddit comments into three categories: `Medical Doctor`, `Veterinarian`, or `Other`. It uses OpenAI's GPT-3.5-turbo for initial classification and then refines the labels using fuzzy matching techniques. The final classification model is trained using a Random Forest classifier with the TF-IDF vectorizer for feature extraction and SMOTE for handling class imbalance.

## Prerequisites

- Python 3.8+
- PostgreSQL database
- OpenAI API key
- pip (Python package installer)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/reddit-comments-classification.git
    cd reddit-comments-classification
    ```

2. Install the required Python packages:
    ```sh
    pip install pandas sqlalchemy tqdm python-dotenv openai fuzzywuzzy python-Levenshtein imbalanced-learn scikit-learn
    ```

3. Ensure you have a `.env` file with your OpenAI API key:
    ```
    OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

### Database Connection

Update the database connection string in the code to connect to your PostgreSQL database:
```python
conn_str = "postgresql://username:password@host/database?options=option&sslmode=require"
```

### Environment Variables

Ensure the environment variables are loaded properly from the `.env` file:
```python
load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```

### Data Sampling

The script samples 800 comments from the Reddit comments dataset for classification:
```python
sample_size = 800
df_sample = df.sample(n=sample_size, random_state=1)
```

### Classification

The comments are classified using OpenAI's GPT-3.5-turbo:
```python
for comment in tqdm(df_sample['comments'], desc="Classifying comments"):
    label = classify_comment(comment)
    labels.append(label)
```

### Data Cleaning

The script uses fuzzy matching to ensure consistent labeling:
```python
df_sample['real_label'] = df_sample['label'].apply(lambda x: get_best_match(x, valid_labels))
```

### Model Training

A Random Forest classifier is trained using TF-IDF features and SMOTE for class balancing:
```python
X = vectorizer.fit_transform(df_sample['comments'])
y = df_sample['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

classifier = RandomForestClassifier(class_weight='balanced', random_state=42)
classifier.fit(X_resampled, y_resampled)
```

## Results

The script outputs classification reports, confusion matrix, and ROC AUC scores to evaluate the model's performance:
```python
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, classifier.predict_proba(X_test), multi_class='ovr'))
```

## Contributing

Contributions/feedbacks are welcome!
