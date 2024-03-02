import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

# Set up NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Initialize stop words
stop_words = set(stopwords.words('spanish'))
ignore_words = ["si", "va", "cómo", "pa", "oh", "yeah", "yah", "yeh", "sé", "así"]
stop_words.update(ignore_words)

# Define directory containing lyric files
cache_dir = '.cache/'

# Function to process text and perform NLP tasks
def process_lyrics(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lyrics = file.read()

    # Tokenize the lyrics
    tokens = word_tokenize(lyrics)

    # Remove stopwords
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Calculate word frequency distribution
    fdist = FreqDist(filtered_tokens)

    # Perform sentiment analysis
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(lyrics)

    return {
        'file_name': os.path.basename(file_path),
        'word_count': len(tokens),
        'unique_words': len(set(tokens)),
        'most_common_words': fdist.most_common(5),
        'sentiment_scores': sentiment_scores
    }

# Iterate through files in cache directory
for file_name in os.listdir(cache_dir):
    file_path = os.path.join(cache_dir, file_name)
    result = process_lyrics(file_path)
    print("Analysis for:", result['file_name'])
    print("Word Count:", result['word_count'])
    print("Unique Words:", result['unique_words'])
    print("Most Common Words:", result['most_common_words'])
    print("Sentiment Scores:", result['sentiment_scores'])
    print("="*50)
