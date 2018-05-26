import nltk
import numpy as np
from nltk.tokenize import word_tokenize #Separates all the words on a sentece
from nltk.stem import WordNetLemmatizer #Removes ing's from words (running and run will be classified as the same word)
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000

def create_lexicon(pos,neg):
    lexicon = []
    for fi in [pos,neg]:
        with open(fi,'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon] #we want the lexicon as efficient as humanly possible
    w_counts = Counter(lexicon)
    #w_counts = {'the':52521,'and':25242}    
    l2 = []
    for w in w_counts:
        if 10000 > w_counts[w] > 10: #If the words are too rare or too common, ignore them
            l2.append(w)
    print(str(len(l2)))
    return l2

def sample_handling(sample,lexicon,classification):
    featureset = []
    with open(sample,'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            featureset.append([features,classification])
    return featureset

def create_fet_sets_n_labels(pos,neg,test_size = 0.1):
    lexicon = create_lexicon(pos,neg)
    features = []
    features += sample_handling('pos.txt',lexicon,[1,0])
    features += sample_handling('neg.txt',lexicon,[0,1])
    random.shuffle(features) #to improve the NN training
    #does tf.argmax([output]) = tf.argmax([expectations]) ? Did we got our prediction right?
    features = np.array(features)
    testing_size = int(test_size*len(features))
    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])
    
    return train_x,train_y,test_x,test_y
    
if __name__ == '__main__':
    train_x,train_y,test_x,test_y = create_fet_sets_n_labels('pos.txt', 'neg.txt')
    with open('sentimet_set.pickle','wb') as f:
        pickle.dump([train_x,train_y,test_x,test_y],f)