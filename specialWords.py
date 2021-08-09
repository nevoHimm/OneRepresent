import pandas as pd
import itertools
from scipy import spatial
from itertools import combinations
import numpy as np


embeddings_dict = {}


def find_subsets(S, m):
    return set(itertools.combinations(S, m))


def find_closest_word(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word_found:
    spatial.distance.euclidean(embeddings_dict[word_found], embedding))


def find_closest_embeddings(word):
    return sorted(embeddings_dict.keys(), key=lambda word_found:
    spatial.distance.euclidean(embeddings_dict[word_found], embeddings_dict[word]))


def find_distance_between_embeddings(embedding_1, embedding_2):
    return spatial.distance.euclidean(embedding_1, embedding_2)


def find_dist_from_set_to_word(word_list, word_2):
    word_vec = [0]*100
    for word in word_list:
        word_vec += embeddings_dict[word]
    return find_distance_between_embeddings(word_vec, embeddings_dict[word_2])


def sub_lists(my_list, n):
    subs = []
    for i in range(0, len(my_list)+1):
        temp = [list(x) for x in combinations(my_list, i)]
        if n > len(temp) > 1 and len(temp) < n:
            subs.extend(temp)
    return subs


def load_dic():
    with open("glove.6B.100d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector


def fix_suffix(word):
    """
    fixes a word with undetectable vocabulary suffix
    :param word: word to fix
    :return: a detectable word for vocabulary
    """
    if len(word) < 1:
        return word
    if word[-2:] == "'s":
        new_word = word[:-2]
    elif len(word) > 3 and word[-3:] == 'ies':
        new_word = word.replace('ies', 'y')
    elif len(word) > 3 and word[-3:] == 'ses':
        new_word = word.replace('ses', 's')
    elif len(word) > 3 and word[-3:] == 'ves':
        new_word = word.replace('ves', 'f')
    elif len(word) > 3 and word[-3:] == 'oes':
        new_word = word.replace('oes', 'o')
    elif len(word) > 3 and word[-2:] == 'es':
        new_word = word.replace('es', 'e')
    elif len(word) > 3 and word[-2:] == 'ity':
        new_word = word.replace('ity', '')
    elif len(word) > 3 and word[-2:] == 'ing':
        new_word = word.replace('ing', '')
    elif word[-1] == 's':
        new_word = word[:-1]
    else:
        return word
    return new_word


def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict


def computeIDF(documents, uniqueWords):
    import math
    N = len(documents)

    idfDict = dict.fromkeys(uniqueWords, 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict


def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf


def main_tfidf():
    data = pd.read_csv('data_out_try.csv', encoding="ISO-8859-1")

    documents = []
    numOfWords = []
    uniqueWords = set()

    for row, col in data.iterrows():
        documents.append(col[1].replace('\n', '').replace('\r', '').split(' '))
        uniqueWords.update(set(documents[-1]))

        numOfWords.append(dict.fromkeys(uniqueWords, 0))
        for word in documents[-1]:
            numOfWords[-1][word] += 1

    tfs = [computeTF(numOfWords[i], documents[i]) for i in range(len(documents))]
    idfs = computeIDF(numOfWords, uniqueWords)
    tfidfs = [computeTFIDF(tfs[i], idfs) for i in range(len(tfs))]
    tf_idf = pd.DataFrame(tfidfs)
    tf_idf.to_csv('TF-IDF.csv', index=False)

    data = pd.read_csv('data_out_try.csv', encoding="ISO-8859-1")

    documents = []
    numOfWords = []
    uniqueWords = set()

    for row, col in data.iterrows():
        documents.append(col[1].split(' '))
        uniqueWords.update(set(documents[-1]))

        numOfWords.append(dict.fromkeys(uniqueWords, 0))
        for word in documents[-1]:
            numOfWords[-1][word] += 1


# main_tfidf()
data = pd.read_csv('data_out_try.csv', encoding="ISO-8859-1")
embeddings_dict = np.load('embeddings_dict.npy', allow_pickle="TRUE").item()

out_put = pd.read_csv('Special_Out.csv', encoding="ISO-8859-1")
special_words = []
documents = []
for row, col in data.iterrows():
    documents.append(col[1].replace('\n', '').replace('\r', '').split(' '))


tf_idf = pd.read_csv('TF-IDF.csv')
i = 0

# for row, col in tf_idf.iterrows():
#     if len(documents[i]) < 6:
#         out_put.at[i, 'Special Words'] = list(col.nlargest(int(len(documents[i])*.8)).keys())
#     elif len(documents[i]) < 11:
#         out_put.at[i, 'Special Words'] = list(col.nlargest(int(len(documents[i])*.6)).keys())
#     elif len(documents[i]) < 17:
#         out_put.at[i, 'Special Words'] = list(col.nlargest(int(len(documents[i])*.4)).keys())
#     else:
#         out_put.at[i, 'Special Words'] = list(col.nlargest(int(len(documents[i])*.25)).keys())
#     i += 1
for m in range(50):
    out_put.at[0, m] = ''

for row, col in tf_idf.iterrows():
    vector = np.zeros((100,), dtype=float)
    words = []
    if len(documents[i]) < 10:
        words = list(col.nlargest(int(len(documents[i])*.8)).keys())
    elif len(documents[i]) < 15:
        words = list(col.nlargest(int(len(documents[i])*.6)).keys())
    elif len(documents[i]) < 20:
        words = list(col.nlargest(int(len(documents[i])*.4)).keys())
    else:
        words = list(col.nlargest(int(len(documents[i])*.25)).keys())
    non_words = list((set(documents[i]) - set(words)) - {''})
    for j in range(len(words)):
        if words[j] != "the" and len(words[j]) > 1 and words[j] in embeddings_dict:
            vector += embeddings_dict[words[j]]
    for k in range(len(non_words)):
        if non_words[k] in embeddings_dict:
            vector -= embeddings_dict[non_words[k]]
    print(words)
    print(non_words)
    print(find_closest_word(vector)[:3])
    print()
    i += 1


out_put.to_csv('Special_Outs.csv', index=False)
