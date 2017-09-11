import numpy as np
from numpy import linalg as LA
import pandas as pd
import pickle
from sklearn import preprocessing
from collections import Counter
from gensim.models import doc2vec
import matplotlib.pyplot as plt


#####################################################################################################
# Data load
#####################################################################################################
### dataset : sonata_data_17479.csv (Additional comment에 해당하는 데이터)
doc2vec_32dim = pickle.load(open('./doc2vec_result_32dim.pickle', 'rb'))
doc2vec_128dim = pickle.load(open('./doc2vec_result_128dim.pickle', 'rb'))
document_index = np.array(pickle.load(open('./document_index.pickle', 'rb')))
sonata_data = pd.read_csv('D://Dropbox//2017_Hyundai//01_data//01_raw_data//sonata_data_17479.csv', delimiter = ',', encoding='latin-1')

mean = []
stddev = []
for i in range(len(doc2vec_32dim)):
    mean.append(np.mean(doc2vec_32dim[i]))
    stddev.append(np.std(doc2vec_32dim[i]))

plt.plot(mean)
plt.plot(stddev)


# Warning 신경 안써도 됨.
doc2vec_32dim_norm = np.array(preprocessing.normalize(doc2vec_32dim[0]))
doc2vec_128dim_norm = np.array(preprocessing.normalize(doc2vec_128dim[0]))

for i in range(len(doc2vec_32dim)-1):
    doc2vec_32dim_norm = np.concatenate((doc2vec_32dim_norm,
                                         preprocessing.normalize(doc2vec_32dim[i+1], norm = 'l2')), axis=0)
    doc2vec_128dim_norm = np.concatenate((doc2vec_128dim_norm,
                                          preprocessing.normalize(doc2vec_128dim[i+1], norm='l2')), axis=0)

with open('./doc2vec_result_32dim_norm.pickle','wb') as mysavedata:
    pickle.dump(doc2vec_32dim_norm, mysavedata)

with open('./doc2vec_result_128dim_norm.pickle','wb') as mysavedata:
    pickle.dump(doc2vec_128dim_norm, mysavedata)