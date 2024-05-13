from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfTransformer
import networkx as nx
import math
from scipy import sparse
import numpy as np
import time
#---------------------------------------Segmentation------------------------------------------#
with open("testData9.txt", 'r', encoding = 'utf-8') as f:
    document=(f.read())

document = document.replace('।', '.')

sentence_tokenizer = PunktSentenceTokenizer()
sentences = sentence_tokenizer.tokenize(document)

print("Sentences:", sentences)
#print(len(sentences))

#-------------------------------Tokenizing-----------------------------------#
with open("testData9.txt", 'r', encoding='utf-8') as f:
    contentRaw = (f.read())

puncAll = []

with open("puncDataAll.txt", 'r', encoding='utf-8') as f:
    puncAll = (f.read())

for i in range(len(puncAll)):
    contentRaw = contentRaw.replace(puncAll[i], '।')

contentRaw = contentRaw.replace('।', '')

contentRaw = contentRaw.replace(' ', '।')

x = contentRaw.split("।")

wordList = []

for i in range(len(x)):
    wordList.append(x[i])

wordList.pop()

print("WordList with Stop Word:", wordList)
#print(len(wordList))
#---------------------------------------Stop Word Removing------------------------------------------#
with open("stopWordList.txt", 'r', encoding='utf-8') as f:
    stopWord = (f.read())

stopWordList = []

stopWordList = stopWord.split("\n")

commonWord = []

for i in range(len(wordList)):
    for j in range(len(stopWordList)):
        if (wordList[i] == stopWordList[j]):
            commonWord.append(wordList[i])

print("Stop Word List in document:", commonWord)
#print(len(commonWord))

for i in range(len(commonWord)):
    if commonWord[i] in wordList:     #edited 12/03/22
        wordList.remove(commonWord[i])

print("WordList without Stop Word:", wordList)
#print(len(wordList))
#---------------------------------------Unique Word List------------------------------------------#
uniqueWordList = []

for i in range(len(wordList)):
    if wordList[i] not in uniqueWordList:
        uniqueWordList.append(wordList[i])

print("Unique Word List:",uniqueWordList)
print(len(uniqueWordList))
#-------------------------------------Sentence tokenizing------------------------------------------#
newSentenceList = []
sentenceToken = []

newSentenceList=sentences

for i in range(len(newSentenceList)):
    newSentenceList[i] = newSentenceList[i].replace('.', ' ')

print("New Sentence List:", newSentenceList)
#print(len(newSentenceList))

sentenceToken = list(map(str.split, newSentenceList))
#print("Sentence Token:", sentenceToken)
#-------------------------------------Duplicate finding------------------------------------------#
duplicates = []
duplicateListIndex = []
for i in range(len(sentenceToken)):
    for item in sentenceToken[i]:
        if sentenceToken[i].count(item) > 1:
            duplicates.append(item)
            duplicateListIndex.append(i)

print("\nDuplicate:", duplicates)
print("\nDuplicate Index:",duplicateListIndex)
#-------------------------------------Matrix generation------------------------------------------#
row= []
column = []
value=[]
checked = 0

for i in range(len(sentenceToken)):
    for k in range(len(sentenceToken[i])):
        for j in range(len(uniqueWordList)):
            if (uniqueWordList[j] == sentenceToken[i][k]) and (sentenceToken[i][k] not in duplicates):
                row.append(i)
                column.append(j)
                value.append('1')
            if (uniqueWordList[j] == sentenceToken[i][k]) and (sentenceToken[i][k] in duplicates) and (checked==0):
                if(i in duplicateListIndex):
                    row.append(i)
                    column.append(j)
                    count=duplicates.count(sentenceToken[i][k])
                    value.append(count)
                    checked=1
                else:
                    row.append(i)
                    column.append(j)
                    value.append('1')
    checked = 0

print("\nRow:", row)
#print(len(row))
print("\nColumn:",column)
#print(len(column))
print("\nValue:",value)
#print(len(value))

row_ind = np.array(row)
col_ind = np.array(column)
data = np.array(value, dtype=float)

mat_coo = sparse.coo_matrix((data, (row_ind, col_ind)))

print("\nSparse matrix indexes:")
print(mat_coo)
print("\nSparse matrix:")
print(mat_coo.toarray())
normalized_matrix = TfidfTransformer().fit_transform(mat_coo)
#----------------------------Normalizing and Similarity matrix generation------------------------------------------#
print("\nNormalized matrix:")
print(normalized_matrix)
similarity_graph = normalized_matrix * normalized_matrix.T

print("\nSimilarity Garph:")
print(similarity_graph.toarray())
#----------------------------Graph Generation and Pagerank implementation------------------------------------------#
nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)       ####from_scipy_sparse_matrix(A, parallel_edges=False, create_using=None, edge_attribute='weight')
scores = nx.pagerank(nx_graph)
print("Scores:\n", scores)
#print(sentences)

ranked = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
print("Rank:", ranked)

#--------------------------Summary Generation---------------------#

with open("testData9.txt", 'r', encoding = 'utf-8') as f:
    content2=(f.read())

content2 = content2.replace('।', '।.')

sentence_tokenizer2 = PunktSentenceTokenizer()
sentences2 = sentence_tokenizer2.tokenize(content2)

for i in range(len(sentences2)):
    sentences2[i] = sentences2[i].replace('।.', '।')

rank = []
copiedScores = []

for i in range(len(scores)):
    copiedScores.append(scores[i])


for i in range(len(scores)):
    a = max(copiedScores)
    for j in range(len(scores)):
        if a == scores[j] and j not in rank:
            rank.append(j)
    copiedScores.remove(a)

z = math.ceil(len(rank) / 4)

finalRank = []

for i in range(z):
    finalRank.append(rank[i])

finalRank.sort()

print("\nSummary:")

for i in range(len(finalRank)):
    a = finalRank[i]
    print(sentences2[a])

'''
############Hybrid Model-2(for write score starts)##########

f = open("textRankScores.txt", "w+")
copiedScores2 = []

for i in range(len(scores)):
    copiedScores2.append(scores[i])

norm = np.linalg.norm(copiedScores2)
normal_array = copiedScores2/norm
print(normal_array)

for i in range(len(normal_array)):
    a = normal_array[i]
    f.write(str(a)+"\n")
f.close()

############Hybrid Model-2(for write score ends)##########
'''

############Accuracy Starts##############
'''
with open("idealSummary5.txt", 'r', encoding='utf-8') as f:
    idealSummary = (f.read())

idealSummary = idealSummary.replace('।', '।.')

idealSummary_sentence_tokenizer = PunktSentenceTokenizer()
idealSummary_sentences = idealSummary_sentence_tokenizer.tokenize(idealSummary)

for i in range(len(idealSummary_sentences)):
    idealSummary_sentences[i] = idealSummary_sentences[i].replace('।.', '।')

print("\nIdeal Summary: ")
for i in range(len(idealSummary_sentences)):
    print(idealSummary_sentences[i])

accuracy = 0
print("\n")
for i in range(len(idealSummary_sentences)):
    for j in range(len(finalRank)):
        a = finalRank[j]
        if(sentences2[a]==idealSummary_sentences[i]):
            accuracy = accuracy + 1
print("\nAccuracy: ", 100*accuracy/len(idealSummary_sentences))
'''