# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from text_process import preprocess_text, generate_word_embeddings, generate_pretrained_word_embeddings


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    text = "France has banned the “recreational” use of TikTok, Twitter, Instagram and other apps on government employees, phones because of concerns about insufficient data security measures.The ban is to come into force immediately, the Ministry of Public Sector Transformation and the Civil Service wrote on Twitter on Friday.."

    # preprocess text
    tokens = preprocess_text(text)

    # generate word embeddings
    frequency_embeddings, frequency_vocab = generate_word_embeddings(tokens, method='tfidf', ngram_range=(1, 2),
                                                                     num_features=1000)
    pretrained_embeddings = generate_pretrained_word_embeddings(tokens, model_name='word2vec', embedding_size=300)

    # print results
    print("Original Text: ", text)
    print("Processed Tokens: ", tokens)
    print("Frequency-based Embeddings: ", frequency_embeddings)
    print("Frequency-based Vocabulary: ", frequency_vocab)
    print("Pretrained Embeddings: ", pretrained_embeddings)


