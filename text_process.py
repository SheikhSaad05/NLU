import re
import nltk
import spacy
import gensim.downloader as api
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer

# download spacy models
nlp = spacy.load('en_core_web_sm')

# download pre-trained models
fasttext_model = api.load('fasttext-wiki-news-subwords-300')
word2vec_model = api.load('word2vec-google-news-300')


# define functions for preprocessing
def clean_text(text):
    # remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def lemmatize_text(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc]


def generate_ngrams(tokens, n=2):
    return list(zip(*[tokens[i:] for i in range(n)]))


def stem_tokens(tokens):
    stemmer = nltk.PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]


def tokenize_text(text):
    return nltk.word_tokenize(text.lower())


# define a function to preprocess text
def preprocess_text(text):
    # clean text
    text = clean_text(text)
    # lemmatize text
    tokens = lemmatize_text(text)
    # generate ngrams
    tokens += generate_ngrams(tokens)
    # stem tokens
    tokens = stem_tokens(tokens)
    # remove stopwords
    tokens = remove_stopwords(tokens)
    # tokenize text
    tokens = tokenize_text(' '.join(tokens))
    return tokens


# define a function to generate word embeddings
def generate_word_embeddings(tokens, method='tfidf', ngram_range=(1, 1), num_features=1000):
    # convert tokens to text for vectorization
    text = ' '.join(tokens)

    # define vectorizer method
    if method == 'bag_of_words':
        vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=num_features)
    elif method == 'cooccurrence_matrix':
        vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=num_features, binary=True)
    elif method == 'hashing_vectorizer':
        vectorizer = HashingVectorizer(n_features=num_features, ngram_range=ngram_range, binary=True)
    elif method == 'one_hot_encoding':
        vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=num_features, binary=True)
    elif method == 'tfidf':
        vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=num_features)

    # generate embeddings
    embeddings = vectorizer.fit_transform([text])

    # return embeddings and vocabulary
    return embeddings.toarray()[0], vectorizer.vocabulary_


# define a function to generate pre-trained word embeddings
def generate_pretrained_word_embeddings(tokens, model_name='word2vec', embedding_size=300):
    if model_name == 'fasttext':
        model = fasttext_model
    elif model_name == 'word2vec':
        model = word2vec_model
    else:
        raise ValueError("Invalid model name. Please use 'fasttext' or 'word2vec'.")

    # remove out-of-vocabulary tokens
    tokens = [token for token in tokens if token in model.vocab]