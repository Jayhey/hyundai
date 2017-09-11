import os
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import normalize
from collections import namedtuple
from gensim.models import doc2vec
import nltk


input_path = "C:\\Users\\Jay\\PycharmProjects\\hyundai\\data"

def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]   # header 제외
    return data

naver_review = read_data(os.path.join(input_path,'ratings_train.txt'))
naver_review = naver_review[:17000]

from konlpy.tag import Twitter
pos_tagger = Twitter()
def tokenize(doc):
    # norm, stem은 optional
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

naver_docs = [(tokenize(row[1]), row[2]) for row in naver_review]

# 잘 들어갔는지 확인
from pprint import pprint
pprint(naver_docs[2])

tokens = [t for d in naver_docs for t in d[0]]

text = nltk.Text(tokens, name='NMSC')

TaggedDocument = namedtuple('TaggedDocument', 'words tags')
# 여기서는 15만개 training documents 전부 사용함
tagged_docs = [TaggedDocument(d, [c]) for d, c in naver_docs]

doc_vectorizer = doc2vec.Doc2Vec(size=128, alpha=0.025, min_alpha=0.025, seed=3434)
doc_vectorizer.build_vocab(tagged_docs)

for epoch in range(100):
    print(epoch)
    doc_vectorizer.train(tagged_docs, epochs=1, total_examples=doc_vectorizer.corpus_count,
                         total_words=len(doc_vectorizer.wv.index2word))
    doc_vectorizer.alpha -= 0.002  # decrease the learning rate
    doc_vectorizer.min_alpha = doc_vectorizer.alpha  # fix the learning rate, no decay


# pprint(doc_vectorizer.most_similar('공포/Noun'))

x = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_docs]

np.save("doc_model_kor_128dim_", x)


### Norm 비교 ###

from numpy import linalg as LA
import matplotlib.pyplot as plt

def norm(x):
    return LA.norm(x)

comment = []
results = []
length = []
for i in range(len(naver_review)):
    comment.append(naver_review[i][1])
    results.append(norm(x[i]))
    length.append(len(naver_review[i][1]))

comment = np.array(comment)
results = np.array(results).astype(np.float)
length = np.array(length).astype(np.float)

np.corrcoef(length,results)

from collections import Counter


labels, values = zip(*Counter(length).items())

indexes = np.arange(len(labels))
width = 1

plt.bar(indexes, values, width)
plt.xticks(indexes + width * 0.5, labels)
plt.show()


c = Counter(length)
plt.bar(c.keys(), c.values())
plt.show()

length_and_norm = np.column_stack((comment,length,results))
df = pd.DataFrame(length_and_norm)

