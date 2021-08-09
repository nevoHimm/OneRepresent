import numpy as np
from scipy import spatial
import itertools
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed, Flatten

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
    with open("glove.6B.300d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector


load_dic()
np.save('embeddings_dict_300.npy', embeddings_dict)
print("hey")
# embeddings_dict = np.load('embeddings_dict.npy', allow_pickle="TRUE").item()
#
# desc = ["heirloom", "indoor", "farms", "grow", "produce"]
#
# sub_descs = sub_lists(desc, 3)
# dists = []
# for sub_desc in sub_descs:
#     dists.append(find_dist_from_set_to_word(sub_desc, "beanstalk"))
#
# print(sub_descs)
# print(sub_descs[dists.index(max(dists))])
# word_vec = [0]*100
# for word in sub_descs[dists.index(max(dists))]:
#     word_vec += embeddings_dict[word]
#
# bad_vec = [0]*100
# bad_words = list(set(desc).difference(set(sub_descs[dists.index(max(dists))])))
# for bad_word in bad_words:
#     bad_vec += embeddings_dict[bad_word]
#
#
# word = [find_closest_word(word_vec - bad_vec)[:20]]

# data = pd.read_csv('try.csv', encoding="ISO-8859-1")
# newdata = pd.DataFrame(columns=["Description", "Words"])
#
# feature_names = data[data.columns[:-1]]
# label_name = data[data.columns[-1]]
#
# j = 0
# for i, row in data.iterrows():
#     if row['Words'] != '0':
#         words = row['Words'].split(',')
#         for word in words:
#             newdata.at[j, 'Description'] = row['Description'].split(' ')
#             newdata.at[j, 'Words'] = word
#             j += 1
#
# newdata.to_csv("maybe.csv", index=False)

embeddings_dict = np.load('embeddings_dict.npy', allow_pickle="TRUE").item()
data = pd.read_csv('try.csv', encoding="ISO-8859-1")
data_after = pd.read_csv('try_after.csv', encoding="ISO-8859-1")


newdata = pd.DataFrame(columns=["Description", "Words"])
newdata_after = pd.DataFrame(columns=["Description", "Words"])

j = 0
for i, row in data.iterrows():
    if row['Words'] != '0':
        words = row['Words'].split(',')
        for word in words:
            blah = []
            if word in embeddings_dict:
                count = 1
                descs = row['Description'].split(' ')
                if len(descs) >= 5:
                    for desc in descs:
                        if desc in embeddings_dict and count < 6:
                            blah += list(embeddings_dict[desc])
                            count += 1
                    if len(blah) == 1500:
                        newdata.at[j, 'Description'] = blah
                        newdata.at[j, 'Words'] = embeddings_dict[word]
                        j += 1

j = 0
for i, row in data_after.iterrows():
    if row['Words'] != '0':
        words = row['Words'].split(',')
        for word in words:
            blah = []
            if word in embeddings_dict:
                count = 1
                descs = row['Description'].split(' ')
                if len(descs) >= 5:
                    for desc in descs:
                        if desc in embeddings_dict and count < 6:
                            blah += list(embeddings_dict[desc])
                            count += 1
                    if len(blah) == 1500:
                        newdata_after.at[j, 'Description'] = blah
                        newdata_after.at[j, 'Words'] = embeddings_dict[word]
                        j += 1

print(newdata_after)
feature_names = newdata[newdata.columns[:-1][0]]
feature_names_after = newdata_after[newdata_after.columns[:-1][0]]
label_name = newdata[newdata.columns[-1]]
label_name_after = newdata_after[newdata_after.columns[-1]]


df = pd.DataFrame([feature_names[0]])
columns = list(df.keys())

for feat in feature_names:
    df = df.append(pd.DataFrame([feat], columns=columns), ignore_index=False)

df_2 = pd.DataFrame([label_name[0]])
columns = list(df_2.keys())
for name in label_name:
    df_2 = df_2.append(pd.DataFrame([name], columns=columns), ignore_index=False)


df_after = pd.DataFrame([feature_names_after[0]])
columns = list(df_after.keys())

for feat in feature_names_after:
    df_after = df_after.append(pd.DataFrame([feat], columns=columns), ignore_index=False)

#
# print(label_train)
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(feat_train, label_train)
# test_pred = knn.predict(feat_test)
# print(test_pred)
# print("kNN model accuracy:", metrics.accuracy_score(label_test, test_pred))

print(df.shape)
print(df_2.shape)
print(df_after.shape)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras import optimizers

model = Sequential()
model.add(Dense(100, input_dim=1500))
model.add(Activation('relu'))
model.add(Dense(300))
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
hist = model.fit(df, df_2, epochs=15, verbose=0, validation_split=0)

y_pred = model.predict(df_after)

model.summary()

words = []
b = 0
for embed in y_pred:
    wordy = find_closest_word(embed)[:10]
    words.append(wordy)
    print(wordy)
    b += 1
df = pd.DataFrame([words]).T
df.to_csv("what.csv", index=False)
