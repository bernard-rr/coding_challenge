import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Load the pre-trained TF-IDF vectorizer and classifier model
vectorizer = joblib.load('vectorizer_u.pkl')
classifier = joblib.load('classifier_model_u.pkl')

st.title('Reddit Comments Classifier')

st.write('Upload a CSV file with a column named `comments` and an optional `label` or `labels` column to classify the comments into categories: "Medical Doctor", "Veterinarian", or "Other". If a `label` or `labels` column is provided, the model will compare the predicted results with the actual labels and display an accuracy score.')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'comments' not in df.columns:
        st.error('The CSV file must contain a column named `comments`.')
    else:
        st.write("Preview of the uploaded CSV file:")
        st.write(df.head())

        # Transform the comments using the loaded TF-IDF vectorizer
        X_new = vectorizer.transform(df['comments'])

        # Predict the labels using the loaded classifier
        df['predicted_label'] = classifier.predict(X_new)

        st.write("Preview of the classified comments:")
        st.write(df.head())

        # Check for the presence of `label` or `labels` columns
        actual_label_column = None
        if 'label' in df.columns:
            actual_label_column = 'label'
        elif 'labels' in df.columns:
            actual_label_column = 'labels'

        if actual_label_column:
            # Convert both actual and predicted labels to lower case for comparison
            df[actual_label_column] = df[actual_label_column].str.lower()
            df['predicted_label'] = df['predicted_label'].str.lower()
            
            # Calculate accuracy
            accuracy = accuracy_score(df[actual_label_column], df['predicted_label'])
            st.write(f'Accuracy of the model: {accuracy:.2f}')

        # Display the final DataFrame with predictions
        st.write("Final DataFrame with predicted labels:")
        st.write(df)
