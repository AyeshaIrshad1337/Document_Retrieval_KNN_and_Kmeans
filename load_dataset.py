import pandas as pd
import os
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import joblib

# Load CSV containing document class information
csv_path = 'document_classes.csv'  # Change to the path of your CSV file
df = pd.read_csv(csv_path)

# Assuming the CSV contains columns 'document_id' and 'class', and your text files are named accordingly.
documents = {}
txt_files_directory = 'ResearchPapers/'  # Change to the directory where your .txt files are located

for index, row in df.iterrows():
    doc_id = row['document_id']
    class_label = row['class']
    with open(os.path.join(txt_files_directory, f'{doc_id}.txt'), 'r', encoding='utf-8', errors='ignore') as file:
        documents[doc_id] = {'text': file.read(), 'class': class_label}

# Convert the dictionary into a DataFrame for further processing
document_df = pd.DataFrame.from_dict(documents, orient='index')

# Load your custom stop words from Stopword.txt
stopword_file_path = 'Stopword-List.txt'  # Change to the path of your Stopword.txt file
with open(stopword_file_path, 'r') as f:
    custom_stopwords = set(f.read().splitlines())

# Download NLTK data
nltk.download('punkt')

# Initialize the stemmer
stemmer = PorterStemmer()

def preprocess_text(text):
    """Function to preprocess and clean text data."""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words and stem
    filtered_tokens = [stemmer.stem(token) for token in tokens if token not in custom_stopwords]
    return ' '.join(filtered_tokens)

# Apply preprocessing to the text column
document_df['processed_text'] = document_df['text'].apply(preprocess_text)

# Vectorize processed text using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(document_df['processed_text'])

# Encode class labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(document_df['class'])


# Save vectorized data
vectorized_data_path = 'vectorized_data.pkl'
joblib.dump(X, vectorized_data_path)

# Save labels
labels_path =  'labels.pkl'
joblib.dump(y, labels_path)

# Save the vectorizer itself for later transformation in other files
vectorizer_path =  'vectorizer.pkl'
joblib.dump(vectorizer, vectorizer_path)

# Save the label encoder for future decoding
label_encoder_path =  'label_encoder.pkl'
joblib.dump(label_encoder, label_encoder_path)
