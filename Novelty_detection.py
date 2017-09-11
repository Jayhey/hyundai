# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import pickle
from sklearn import mixture
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

from collections import Counter
from gensim.models import doc2vec
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics

#####################################################################################################
# Load Doc2Vec embedding result
#####################################################################################################

# Normalize x
# doc2vec_32dim = pickle.load(open('./doc2vec_result_32dim.pickle', 'rb'))
# doc2vec_128dim = pickle.load(open('./doc2vec_result_128dim.pickle', 'rb'))
# document_index = np.array(pickle.load(open('./document_index.pickle', 'rb')))

# Normalize O
doc2vec_32dim = pickle.load(open('./doc2vec_result_32dim_norm.pickle', 'rb'))
doc2vec_128dim = pickle.load(open('./doc2vec_result_128dim_norm.pickle', 'rb'))
document_index = np.array(pickle.load(open('./document_index.pickle', 'rb')))




sonata_data = pd.read_csv('./sonata_data_17479.csv', delimiter = ',', encoding='latin-1')
#comment 정보
sonata_comments = sonata_data[['Comment']]
#space bar를 '로 바꾸기, 양옆 여백 제거, 소문자화
sonata_space2upper = []
for i in range(len(sonata_comments)):
    tmp = [str(w).replace('n t ', 'n\'t ').replace('t s ', 't\'s ').replace(' d ', '\'d ').replace(' m ', '\'m ').replace(' ve ', "\'ve ").strip().lower() for w in sonata_comments.iloc[i]]
    sonata_space2upper.append(tmp)
#데이터셋에서 index까지 만들어서 다루기. 나중에 연결할 때 용이하게 하기 위함
sonata_corpus = np.array([[ind,value[0]] for ind, value in enumerate(np.array(sonata_space2upper))])



#####################################################################################################
# Models
#####################################################################################################
########################################
# Gaussian Mixture model
########################################
components_list = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
cov_list = np.array(['spherical', 'tied', 'diag', 'full'])

gmm_result = []
for comp in components_list:
    for cov in cov_list:
        information = 'n_components_{comp}__cov_{cov}'.format(comp=comp, cov=cov)
        print(information)

        #32-dim
        gmm = mixture.GMM(n_components=comp, covariance_type=cov)
        gmm_32 = gmm.fit(X=doc2vec_32dim)
        aic_32 = gmm_32.aic(doc2vec_32dim)
        bic_32 = gmm_32.bic(doc2vec_32dim)
        #128-dim
        gmm = mixture.GMM(n_components=comp, covariance_type=cov)
        gmm_128 = gmm.fit(X=doc2vec_128dim)
        aic_128 = gmm_128.aic(doc2vec_128dim)
        bic_128 = gmm_128.bic(doc2vec_128dim)

        gmm_result.append([information, aic_32, bic_32, aic_128, bic_128])

gmm_result = pd.DataFrame(gmm_result)
gmm_result.to_csv('./gmm_result.csv')






########################################
# K-Nearest Neighbor
########################################
neighbors_list = [3, 5, 7, 9, 11, 13, 15]
data_list = {'doc2vec_32dim': doc2vec_32dim, 'doc2vec_128dim': doc2vec_128dim}

def KNN(data, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data)
    distances, indices = nbrs.kneighbors(data)
    dist_mean = np.mean(distances[:, 1:], axis=1)

    return(dist_mean)

def Top_and_low(knn, num, hist=True):
    top = np.argsort(knn)[-num:]
    low = np.argsort(knn)[:num]
    rows = np.r_[low, top]
    scores = knn[rows]
    docs = document_index[rows].astype(str)
    ROWS = np.empty(shape=[6, 3], dtype=object)
    for ind, val in enumerate(docs):
        row = np.c_[sonata_corpus[sonata_corpus[:,0] == val], scores[ind]][0]
        ROWS[ind] = row

    if hist:
        plt.hist(knn, bins=100)

    return(ROWS)

OUTPUT = np.empty([0, 4], dtype=object)
for key in data_list.keys():
    # key = 'doc2vec_32dim'
    data = data_list.get(key)
    for k in neighbors_list:
        print(key, '||', k)
        info = str(key) + '_k' + str(k)
        mean_knn = KNN(data, k)
        tmp = Top_and_low(mean_knn, 3, False)
        output = np.c_[np.repeat(key, 6), tmp]
        OUTPUT = np.concatenate([OUTPUT, output])


pd.DataFrame(OUTPUT).to_csv('./TopAndLowSentences.csv')
