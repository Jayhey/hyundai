import numpy as np
from numpy import linalg as LA
import pandas as pd
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

from collections import Counter
from gensim.models import doc2vec



#####################################################################################################
# Data load
#####################################################################################################
### dataset : sonata_data_17479.csv (Additional comment에 해당하는 데이터)
doc2vec_32dim = pickle.load(open('./doc2vec_result_32dim.pickle', 'rb'))
doc2vec_128dim = pickle.load(open('./doc2vec_result_128dim.pickle', 'rb'))
document_index = np.array(pickle.load(open('./document_index.pickle', 'rb')))
sonata_data = pd.read_csv('D://Dropbox//2017_Hyundai//01_data//01_raw_data//sonata_data_17479.csv', delimiter = ',', encoding='latin-1')

#comment 정보
sonata_comments = sonata_data[['Comment']]
#space bar를 '로 바꾸기, 양옆 여백 제거, 소문자화
sonata_space2upper = []
for i in range(len(sonata_comments)):
    tmp = [str(w).replace('n t ', 'n\'t ').replace('t s ', 't\'s ').replace(' d ', '\'d ').replace(' m ', '\'m ').replace(' ve ', "\'ve ").strip().lower() for w in sonata_comments.iloc[i]]
    sonata_space2upper.append(tmp)
#데이터셋에서 index까지 만들어서 다루기. 나중에 연결할 때 용이하게 하기 위함
sonata_corpus = np.array([[ind,value[0]] for ind, value in enumerate(np.array(sonata_space2upper))])
#delete duplicated rows
_, indices = np.unique(sonata_corpus[:,1], return_index=True)
sonata_corpus = sonata_corpus[indices]
#delete 'na'
na_list = np.where(sonata_corpus[:,1] == 'na')[0]
sonata_corpus = np.delete(sonata_corpus, na_list, axis=0)
#sonata_corpus[0]과 sonata_space2upper[14609]가 제대로 매칭이 되어 있음

#최종 데이터 개수 : 17479 -> 17049



#####################################################################################################
# Preprocessing
#####################################################################################################
### Tokenization
tokenizer = RegexpTokenizer(r'\w+')
def tokenize(data):
    tok_data = list(map(lambda x: tokenizer.tokenize(x), data))
    return tok_data
tok_sonata = tokenize(sonata_corpus[:,1])

### Lemmatizer
lemmatizer = WordNetLemmatizer()
def lemma(data):
    tmp = []
    for i in range(len(data)):
        if i % 100:
            print(i)
        doc = data[i]
        try:
            tmp.append([lemmatizer.lemmatize(word) for word in doc])
        except IndexError:
            print('IndexError')
            pass
    return tmp
Lemma_sonata = lemma(tok_sonata)


#####################################################################################################
# Load Doc2Vec embedding result
#####################################################################################################

def norm(x):
    return LA.norm(x)

comment = []
results = []
length = []
for i in range(len(document_index)):
    comment.append(sonata_comments.ix[document_index[i]])
    results.append(norm(doc2vec_32dim[i]))
    length.append(len(Lemma_sonata[i]))

comment = np.array(comment)
results = np.array(results).astype(np.float)
length = np.array(length).astype(np.float)

length_and_norm = np.column_stack((document_index, comment,length,results))
df = pd.DataFrame(length_and_norm)
df.to_csv("correlation.csv")