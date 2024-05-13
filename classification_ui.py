import streamlit as st
import joblib
import os
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import numpy as np
import pandas as pd
from load_dataset import preprocess_text
# Load pre-saved data and models
vectorized_data_path = 'vectorized_data\\vectorized_data.pkl'
labels_path = 'vectorized_data\\labels.pkl'
vectorizer_path = 'vectorized_data\\vectorizer.pkl'
label_encoder_path = 'vectorized_data\\label_encoder.pkl'

X = joblib.load(vectorized_data_path)
y = joblib.load(labels_path)
vectorizer = joblib.load(vectorizer_path)
label_encoder = joblib.load(label_encoder_path)

# Load original document IDs
document_ids = pd.read_csv('document_classes.csv')['document_id']

# Ensure document_ids aligns with labels
assert len(document_ids) == len(y), "Mismatch between document IDs and labels."

# Get all unique labels
all_labels = np.unique(y)

# Streamlit app
st.title('Document Classification and Clustering App')
st.write('This application showcases the results of KNN and KMeans on document classification.')
# Choose model type
model_type = st.selectbox('Select Model', ['KNN', 'KMeans'])
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(splitter.split(X, y))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
train_doc_ids = document_ids.iloc[train_idx].values
test_doc_ids = document_ids.iloc[test_idx].values

if model_type == 'KNN':
    st.subheader('KNN Classifier Results')
    # Initialize and train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Predict on the training set
    y_train_pred = knn.predict(X_train)

    # Predict on the test set
    y_test_pred = knn.predict(X_test)

    # Confusion matrix for training data
    cm_train = confusion_matrix(y_train, y_train_pred, labels=all_labels)
    fig, ax = plt.subplots()
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.inverse_transform(all_labels),
                yticklabels=label_encoder.inverse_transform(all_labels))
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('KNN Confusion Matrix (Training Data)')
    st.pyplot(fig)

    # Confusion matrix for test data
    cm_test = confusion_matrix(y_test, y_test_pred, labels=all_labels)
    fig, ax = plt.subplots()
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.inverse_transform(all_labels),
                yticklabels=label_encoder.inverse_transform(all_labels))
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('KNN Confusion Matrix (Test Data)')
    st.pyplot(fig)

    # Classification report for training data
    report_train = classification_report(y_train, y_train_pred, labels=all_labels, target_names=label_encoder.inverse_transform(all_labels))
    st.text('Classification Report (Training Data):')
    st.text(report_train)

    # Classification report for test data
    report_test = classification_report(y_test, y_test_pred, labels=all_labels, target_names=label_encoder.inverse_transform(all_labels))
    st.text('Classification Report (Test Data):')
    st.text(report_test)

    # Accuracy for test data
    accuracy_test = accuracy_score(y_test, y_test_pred)
    st.text(f'Accuracy (Test Data): {accuracy_test:.4f}')

elif model_type == 'KMeans':
    st.subheader('KMeans Clustering Results')
    # Get input for k
    k = st.slider('Select number of clusters (k)', min_value=2, max_value=10, value=5)

    # Train KMeans clustering model
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)

    # Evaluate the clusters
    cluster_labels = kmeans.labels_
    cm = confusion_matrix(y, cluster_labels)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(k),
                yticklabels=all_labels)
    ax.set_xlabel('Cluster Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'KMeans Confusion Matrix (k={k})')
    st.pyplot(fig)

    # Display clusters as a DataFrame
    df_clusters = pd.DataFrame({'Document ID': document_ids, 'True Label': label_encoder.inverse_transform(y), 'Cluster': cluster_labels})
    st.dataframe(df_clusters)

    # Visualize the distribution of documents per cluster
    fig, ax = plt.subplots()
    sns.countplot(x='Cluster', data=df_clusters, palette='Set3', ax=ax)
    ax.set_title('Document Distribution per Cluster')
    st.pyplot(fig)

    # Purity calculation
    def calculate_purity(cm):
        return np.sum(np.amax(cm, axis=0)) / np.sum(cm)

    purity = calculate_purity(cm)
    silhouette = silhouette_score(X, cluster_labels)
    rand_index = adjusted_rand_score(y, cluster_labels)

    st.text(f'Purity: {purity:.4f}')
    st.text(f'Silhouette Score: {silhouette:.4f}')
    st.text(f'Rand Index: {rand_index:.4f}')


# Text input for new document classification
st.subheader("Classify a new document")
user_input = st.text_area("Enter the text of the document you wish to classify:", height=150)
if st.button('Classify Document'):
    if user_input:
        # Assuming the preprocessing function and necessary resources are available
        processed_text = preprocess_text(user_input)  # The function preprocess_text needs to be defined or imported
        vectorized_text = vectorizer.transform([processed_text])
        
        if model_type == 'KNN':
            prediction = knn.predict(vectorized_text)
        elif model_type == 'KMeans':
            prediction = kmeans.predict(vectorized_text)
        
            cluster_number = kmeans.predict(vectorized_text)
            st.write(f'The document belongs to cluster number: **{cluster_number[0]}**')
    else:
        st.write("Please enter some text to classify.")


