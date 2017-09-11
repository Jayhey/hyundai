# -*- coding:utf-8 -*-
#####################################################################################################
# Import modules
#####################################################################################################
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import pickle
# from stop_words import get_stop_words
# import re
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from collections import Counter
# from nltk.stem.porter import PorterStemmer



#####################################################################################################
# Data load
#####################################################################################################
### dataset : sonata_data_17479.csv (Additional comment에 해당하는 데이터)
sonata_data = pd.read_csv('../../01_data/01_raw_data/sonata_data_17479.csv', delimiter = ',', encoding='latin-1')
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

# ### Stemming
# # 오류나는 행들이 조금 있음
# p_stemmer = PorterStemmer()
# def stem(data):
#     tmp = []
#     for i in range(len(data)):
#         if i % 100:
#             print(i)
#         doc = data[i]
#         try:
#             tmp.append([p_stemmer.stem(word) for word in doc])
#         except IndexError:
#             print('IndexError')
#             pass
#     return tmp
# stem_sonata = stem(tok_sonata)

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

index=[]
# merge "s" and "t" to "n't" and "it's" 등등...
# 전처리 후 lemmatize를 거치면서 떨어지는데, 여기에 '를 붙여주기. 조동사 류 라는것을 딱 보고 확인할수 있도록 하기 위함
for i in range(len(Lemma_sonata)):
    ind = np.where(np.in1d(Lemma_sonata[i], np.array(['t', 's', 'd', 'm', 've'])))[0]

    #'t', 's', 'd', 'm', 've' 중에 하나라도 있다면 바꿔주기
    if len(ind) > 0:
        for k in ind:
            Lemma_sonata[i][k] = '\'' + Lemma_sonata[i][k]
    else:
        pass

    if i % 1000 == 0:
        print("Finished ", i, "th record!!!!")


# Doc2Vec, Word2Vec류의 단어 순서가 중요한 데이터의 경우, stopwords 처리를 먼저 해주면 단어 순서가 망가지므로 적용하지 않고 진행
# ### stopwords
# en_stop = get_stop_words('en')
# def removeStopwords(data):
#     i=0
#     corpus = []
#     for doc in data:
#         print(i)
#         corpus.append([word for word in doc if word not in en_stop])
#         i += 1
#     return corpus
# sonata_stop = removeStopwords(Lemma_sonata)
# sonata_stop

# 중복제거는 위에서 진행해줌
# # remove duplicated
# sonata_stop_drop = []
# for i in Lemma_sonata:
#   if i not in sonata_stop_drop:
#       sonata_stop_drop.append(i)


### count words
from collections import Counter
unlist_corpus = sum(Lemma_sonata, [])
count_words = Counter(unlist_corpus)

import operator
# operator.itemgetter(*items)
# g = itemgetter(2, 5, 3), the call g(r) returns (r[2], r[5], r[3]).
sorted_words = sorted(count_words.items(), key=operator.itemgetter(1)) #item중에서 1번째 것들만 모두 가져옴. R에서 apply(data, '[', 1)과 유사함
sorted_words.reverse()

#data save
# pd.DataFrame.to_csv(pd.DataFrame(sorted_words), './sorted_words.csv')


#####################################################################################################
# Doc2Vec
#####################################################################################################
from collections import namedtuple
from gensim.models import doc2vec

#128차원 공간에 임베딩, dm=1(pv-dm 적용), dm_concat=1(1-layer의 hidden node 구성시 word vector와 paragraph vector concatenate. 논문상에서 average보다 concatenate가 좋다고 얘기함), min_count=2(최소 2회 이상 출현 단어)
config = {'size':128, 'dm':1, 'dm_concat':1, 'min_count':2, 'window':5, 'workers':4}

#namedtuple 형으로 만들어줘야 doc2vec을 실행할 수 있다.
sonata_tagged_document = namedtuple('sonata_data', ['words', 'tags'])
#namedtuple 만드는 과정
# tag자리에 document index가 들어가야 하는데, 이것이 매칭이 잘 안되있으므로, 이를 바로잡으려고 for문으로 직접 돌림
# sonata_tagged_tr_document = [sonata_tagged_document(words, [tags]) for tags, words in enumerate(Lemma_sonata)]
sonata_tagged_tr_document = []
document_index = []
for ind, words in enumerate(Lemma_sonata):
    tags = int(sonata_corpus[ind,0])
    document_index.append(tags)
    sonata_tagged_tr_document.append(sonata_tagged_document(words, [ind]))

#doc2vec 객체 생성 및 하이퍼파라미터 설정
doc_model = doc2vec.Doc2Vec(**config)
doc_model.build_vocab(sonata_tagged_tr_document)
# #임베딩 할 총 문서의 개수와 단어의 개수
doc_num = doc_model.docvecs.count
word_num = len(doc_model.wv.index2word)

#training : 100 epoch
for epoch in range(100):
    print(epoch)
    #training document로 학습
    doc_model.train(sonata_tagged_tr_document, epochs=1, total_examples=doc_num, total_words=word_num)
    #learning rate decay
    doc_model.alpha -= 0.002
    #최소
    doc_model.min_alpha = doc_model.alpha

# save doc2vec model
doc_model.save('./doc_model_sonata_128dim')

# document embedding
sonata_doc2vec = np.asarray(doc_model.docvecs)

# save
with open('./doc2vec_result_128dim.pickle','wb') as mysavedata:
    pickle.dump(sonata_doc2vec, mysavedata)

#document index
with open('./document_index.pickle','wb') as mysavedata:
    pickle.dump(document_index, mysavedata)
