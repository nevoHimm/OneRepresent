# from keras.preprocessing.text import one_hot
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Flatten
# from keras.layers.embeddings import Embedding
# from nltk.tokenize import word_tokenize
# import pandas as pd
# # import nltk
# # nltk.download('punkt')
#
# data = pd.read_csv('keras.csv', encoding="ISO-8859-1")
# corpus = list(data[data.columns[0]])
# sentiments = list(data[data.columns[1]])
#
# # all_words = []
# # for sent in corpus:
# #     tokenize_word = word_tokenize(sent)
# #     for word in tokenize_word:
# #         all_words.append(word)
# #
# # unique_words = set(all_words)
# # print(len(unique_words))
#
# vocab_length = 100000
#
# embedded_sentences = [one_hot(sent, vocab_length) for sent in corpus]
# embedded_sentiments = [one_hot(str(word), vocab_length)[0] for word in sentiments]
#
# word_count = lambda sentence: len(word_tokenize(sentence))
# longest_sentence = max(corpus, key=word_count)
# length_long_sentence = len(word_tokenize(longest_sentence))
# padded_sentences = pad_sequences(embedded_sentences, length_long_sentence, padding='post')
# print(padded_sentences)
# print(length_long_sentence)
# model = Sequential()
# model.add(Embedding(vocab_length, 64, input_length=length_long_sentence))
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam', loss='kullback_leibler_divergence', metrics=['acc'])
# print(model.summary())
# model.fit(padded_sentences, embedded_sentiments, epochs=50, verbose=1)
# loss, accuracy = model.evaluate(padded_sentences, embedded_sentiments, verbose=0)
# print('Accuracy: %f' % (accuracy*100))
#
# data_test = pd.read_csv('‏‏keras_test.csv', encoding="ISO-8859-1")
# corpus_test = list(data[data.columns[0]])
# embedded_sentences1 = [one_hot(sent, vocab_length) for sent in corpus_test]
# word_count1 = lambda sentence: len(word_tokenize(sentence))
# longest_sentence1 = max(corpus, key=word_count1)
# length_long_sentence1 = len(word_tokenize(longest_sentence1))
# padded_sentences = pad_sequences(embedded_sentences1, length_long_sentence1, padding='post')
#
# print(model.predict(padded_sentences))

from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
import numpy as np
import pandas as pd

embeddings_dict = np.load('embeddings_dict.npy', allow_pickle="TRUE").item()
data = pd.read_csv('try.csv', encoding="ISO-8859-1")

newdata = pd.DataFrame(columns=["Description", "Words"])

docs = []
labels = []

for i, row in data.iterrows():
    if row['Words'] != '0':
        words = row['Words'].split(',')
        for word in words:
            if word in embeddings_dict:
                docs.append(row['Description'])
                labels.append(word)


# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)
# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)
# load the whole embedding into memory
embeddings_index = dict()
f = open('glove.6B.300d.txt', encoding="ISO-8859-1")
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:])
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 300))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
# define model

print(embedding_matrix[0])
# model = Sequential()
# e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=4, trainable=False)
# model.add(e)
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))
# # compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# # summarize the model
# print(model.summary())
# # fit the model
# model.fit(np.array(padded_docs), labels, epochs=50, verbose=0)
# # evaluate the model
# loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
# print('Accuracy: %f' % (accuracy*100))