import nltk
from nltk.corpus import twitter_samples
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.classify import NaiveBayesClassifier
import random

# Download necessary NLTK data (run this once)
# nltk.download('twitter_samples')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')

# Load sample tweets
twitter_samples.fileids()
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

# --- Text Preprocessing ---
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_tweet(tweet):
    tokens = tokenizer.tokenize(tweet)
    processed_tokens = []
    for token in tokens:
        # Remove URLs, hashtags, and mentions (basic)
        if token.startswith('http') or token.startswith('#') or token.startswith('@'):
            continue
        # Lemmatize and remove stop words
        lemma = lemmatizer.lemmatize(token)
        if lemma not in stop_words and len(lemma) > 1:
            processed_tokens.append(lemma)
    return processed_tokens

# Preprocess all tweets and create feature sets
all_tweets = [(preprocess_tweet(tweet), 'positive') for tweet in positive_tweets] + \
             [(preprocess_tweet(tweet), 'negative') for tweet in negative_tweets]

random.shuffle(all_tweets)

def extract_features(tweet_tokens):
    features = {}
    for word in tweet_tokens:
        features[f'contains({word})'] = True
    return features

feature_sets = [(extract_features(tweet), sentiment) for tweet, sentiment in all_tweets]

# Split data into training and testing sets
train_size = int(len(feature_sets) * 0.8)
train_set, test_set = feature_sets[:train_size], feature_sets[train_size:]

# --- Model Training ---
classifier = NaiveBayesClassifier.train(train_set)

# --- Model Evaluation ---
accuracy = nltk.classify.accuracy(classifier, test_set) * 100
print(f"Accuracy: {accuracy:.2f}%")

# --- Prediction ---
def predict_sentiment(text):
    processed_text = preprocess_tweet(text)
    features = extract_features(processed_text)
    return classifier.classify(features)

# Example usage
print("\n--- Predictions ---")
print(f"'I love this product! It's amazing.' -> {predict_sentiment('I love this product! It\'s amazing.')}")
print(f"'This is the worst experience ever.' -> {predict_sentiment('This is the worst experience ever.')}")
print(f"'The weather is okay today.' -> {predict_sentiment('The weather is okay today.')}")
