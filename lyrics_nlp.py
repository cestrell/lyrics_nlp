import os
import csv
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from syrics.api import Spotify

# Constants
CACHE_DIR = '.cache/'
SP_DC_KEY_FILE = "SP_DC_KEY.txt"
DATASET_CSV = "labeled_lyrics.csv"

# Initialize Syrics
with open(SP_DC_KEY_FILE, "r", encoding='utf-8') as file:
    sp_dc = file.read().strip()
syrics = Spotify(sp_dc)

# Initialize stop words
stop_words = set(stopwords.words('spanish'))
ignore_words = ["si", "va", "cómo", "pa", "oh", "yeah", "yah", "yeh", "sé", "así"]
stop_words.update(ignore_words)

# Define playlist URLs
training_playlists = {
    "arriba": "https://open.spotify.com/playlist/288cWRayDQcwkUSQKDXEqD",
    "sensual": "https://open.spotify.com/playlist/2eRBHFn5gnzRl5sajXyYyX",
    "enamorado": "https://open.spotify.com/playlist/7J3T3yk3adjpMcX2gLIA6x",
    "belico": "https://open.spotify.com/playlist/5BMDtne5hSB351LVWKQdrG",
    "triste": "https://open.spotify.com/playlist/3uoIOMmzWNvGrZwcpdsiA1"
    }

# Record tracks in each training playlist
playlist_tracks = {}

# Retrieve tracks in each training playlist
def fetch_tracks():
    for playlist_name, playlist_url in training_playlists.items():
        playlist_data = syrics.playlist(playlist_url)
        
        # Append tracks to corresponding playlists
        playlist_tracks[playlist_name] = [track['track']['id'] for track in playlist_data['tracks']['items']]

# Create bag of words model
def extract_lyrics_features(lyrics):
    tokens = word_tokenize(lyrics.lower())
    # I could use list expansion but it looks messy
    filtered_tokens = []
    for token in tokens:
        if token.isalpha() and token not in stop_words:
            filtered_tokens.append(token)

    # Feature engineering: generate bigrams/trigrams for increased accuracy
    bigrams = list(ngrams(filtered_tokens, 2))
    trigrams = list(ngrams(filtered_tokens, 3))

    bi = [' '.join(bigram) for bigram in bigrams]
    tri = [' '.join(trigram) for trigram in trigrams]
    
    features = filtered_tokens + bi + tri

    # Feature representation of lyrics for sentiment analysis
    return dict((word, True) for word in features)

def read_lyrics(track_id, playlist_name):
    lyrics_file = os.path.join(CACHE_DIR, f"{track_id}")
    if os.path.exists(lyrics_file):
        with open(lyrics_file, 'r', encoding='utf-8') as file:
            lyrics = file.read()
            sentences = sent_tokenize(lyrics)
            # Append each sentence as a separate entry in the dataset
            dataset_entries = []
            for sentence in sentences:
                lyrics_features = extract_lyrics_features(sentence)
                dataset_entries.append((lyrics_features, playlist_name))
            return dataset_entries
            # Deprecated; no advanced tokenization
            # lyrics = file.read().strip()
            # return lyrics
    else:
        # Hack for no lyrics, use category in place of lyrics
        format_name = extract_lyrics_features(playlist_name)
        return [(format_name, playlist_name)]

# Prepare data entries for csv
def construct_dataset_entries():
    dataset = []
    for playlist_name in playlist_tracks:
        for track_id in playlist_tracks[playlist_name]:
            # Read lyrics and extract features
            lyrics_entries = read_lyrics(track_id, playlist_name)
            for lyrics_features, playlist_name in lyrics_entries:
                dataset.append((lyrics_features, playlist_name))

            # Deprecated; no advanced tokenization
            # lyrics = read_lyrics(track_id, playlist_name)
            # lyrics_features = extract_lyrics_features(lyrics)
            # dataset.append((lyrics_features, playlist_name))
    return dataset

def dataset_to_csv(dataset):
    with open(DATASET_CSV, 'w', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['lyric', 'sentiment'])
            writer.writerows(dataset)

def create_dataset():
    fetch_tracks()
    dataset = construct_dataset_entries()
    dataset_to_csv(dataset)
    return dataset

def predict_sentiment(lyrics):
    # Load the saved classifier from a file
    with open('classifier.pickle', 'rb') as f:
        classifier = pickle.load(f)

    lyrics_features = extract_lyrics_features(lyrics)
    return classifier.classify(lyrics_features)

def train_classifier():
    # Use playlist URLs to generate training dataset
    dataset = create_dataset()

    # Divide into training and testing
    split_ratio = 0.8
    train_size = int(len(dataset) * split_ratio)
    train_set, test_set = dataset[:train_size], dataset[train_size:]
    
    # Train Naive Bayes classifier
    classifier = NaiveBayesClassifier.train(train_set)

    # Determine accuracy on testing set
    accuracy_test = accuracy(classifier, test_set)
    print('Accuracy:', accuracy_test)
    
    # Save the trained classifier to a file
    with open('classifier.pickle', 'wb') as f:
        pickle.dump(classifier, f)

def classify_test_lyrics():
    TEST_DIR = "test_belicos/"
    for track_id in os.listdir(TEST_DIR):
        track_path = os.path.join(TEST_DIR, track_id)
        if os.path.isfile(track_path):
            with open(track_path, 'r', encoding='utf-8') as file:
                lyrics = file.read()
                sentiment = predict_sentiment(lyrics)
                print(f"{track_id}: {sentiment}")

def display_menu():
    message = """\nWelcome to LyricsNLP:
    1. Train classifier
    2. Classify test lyrics
    3. Exit
    """
    print(message)

def main():
    display_menu()
    choice = input("Choice: ")
    choice = int(choice) if choice.isnumeric() else 0
    if choice == 1:
        train_classifier()
    elif choice == 2:
        classify_test_lyrics()
    else:
        exit()

if __name__ == "__main__":
    main()