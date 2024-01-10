import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset (replace "imdb_dataset.csv" with the actual file name)
df = pd.read_csv("imdb_dataset.csv")

# Display the first few rows of the original dataset
print("Original Dataset:")
print(df.head())

# Data Preprocessing
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenization and removal of stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

# Apply text preprocessing to the 'review' column
df['processed_review'] = df['review'].apply(preprocess_text)

# Display the first few rows of the preprocessed dataset
print("\nPreprocessed Dataset:")
print(df[['review', 'processed_review']].head())

# Save the preprocessed dataset to a new CSV file
df.to_csv("preprocessed_imdb_dataset.csv", index=False)
