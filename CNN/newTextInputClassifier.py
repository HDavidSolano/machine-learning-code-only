import nltk
import numpy as np
from nltk.tokenize import word_tokenize,RegexpTokenizer #Separates all the words on a sentece
from nltk.stem import WordNetLemmatizer #Removes ing's from words (running and run will be classified as the same word)
import random
import pickle
from collections import Counter
import os
import math as m
import shutil
import re

rootdir = 'C:\\Users\\David\\Desktop\\20_newsgroups' # Source folder with all the examples to use to label and classify
training_dir = 'C:\\Users\\David\\Desktop\\MachineLearning\\Training'
validation_dir = 'C:\\Users\\David\\Desktop\\MachineLearning\\Validation'
training_frac = 0.1

''' First, we separate the training and validation sets, WITHOUT ANY MODIFICATION OR CARACTERIZATION of the training set'''

def split_data(rootdir,training_dir,validation_dir):
    dir_num = 0
    for _, dirs, _ in os.walk(rootdir):
        for adir in dirs:
            print('working on ',adir)
            train_count = 0
            sub_category = os.path.join(rootdir, adir)  # Here we got the directory we want, and adir has the label we want 
            alist = os.listdir(sub_category)
            number_files = len(alist)
            print('todo ',str(number_files))
            training_portion = m.floor((1-training_frac)*number_files)  # Preparing the fraction of examples which will go to training or validation
            for  files in alist:
                if train_count < training_portion:
                    new_name = files+str(dir_num)
                    folderPath = os.path.join(sub_category, files)
                    newfolderPath = os.path.join(training_dir, new_name)
                    shutil.copy(folderPath, newfolderPath)
                else:
                    new_name = files+str(dir_num)
                    folderPath = os.path.join(sub_category, files)
                    newfolderPath = os.path.join(validation_dir, new_name)
                    shutil.copy(folderPath, newfolderPath)
                train_count+=1
            dir_num+=1
lemmatizer = WordNetLemmatizer()
hm_lines = 10000000
def create_dict(training_dir):
    
    tokenizer = RegexpTokenizer(r'\w+')
    lexicon = []
    alist = os.listdir(training_dir)
    for files in alist:
        file_path = os.path.join(training_dir, files)
        with open(file_path,'r') as f:
            contents = f.readlines()
            email_contents = False      #Flag that will signal when we are really reading the email contents
            start_tag = 'Lines:'
            for l in contents[:hm_lines]:
                if email_contents == True and not('Newsgroups:' in l or 'Path:' in l or 'From:' in l or 'Subject:' in l or 'Organization:' in l or 'Distribution:' in l or 'Date:' in l or 'Message-ID:' in l or 'Xref:' in l or 'Sender:' in l or 'References:' in l):
                    all_words = tokenizer.tokenize(l.lower())
                    lexicon += list(all_words)
                if start_tag in l and email_contents == False:
                    email_contents = True
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon] #we want the lexicon as efficient as humanly possible
    w_counts = Counter(lexicon)
    l2 = []
    min_consider = 100
    max_consider = 8000
    for w in w_counts:
        if max_consider > w_counts[w] > min_consider: #If the words are too rare or too common, ignore them
            l2.append(w)
    print(str(len(l2)))
    return l2
def sample_handling(sample,lexicon):
    featureset = []
    tokenizer = RegexpTokenizer(r'\w+')
    label_map =  {'alt.atheism':[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  'comp.graphics':[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  'comp.os.ms-windows.misc':[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  'comp.sys.ibm.pc.hardware':[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  'comp.sys.mac.hardware':[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  'comp.windows.x':[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  'misc.forsale':[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  'rec.autos':[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                  'rec.motorcycles':[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                  'rec.sport.baseball':[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                  'rec.sport.hockey':[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                  'sci.crypt':[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                  'sci.electronics':[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                  'sci.med':[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                  'sci.space':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                  'soc.religion.christian':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                  'talk.politics.guns':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                  'talk.politics.mideast':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                  'talk.politics.misc':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                  'talk.religion.misc':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]}
    with open(sample,'r') as f:
        contents = f.readlines()
        email_contents = False      #Flag that will signal when we are really reading the email contents
        start_tag = 'Lines:'
        classifier_tag = 'Newsgroups:'
        classified = False
        features = np.zeros(len(lexicon))
        for l in contents[:hm_lines]:
            if classifier_tag in l:
                groups = re.findall(r'[^,;\s]+', l)
                for a_group in groups:
                    if a_group in label_map and classified == False:
                        the_label = label_map[a_group]
                        classified = True
                if classified == False:
                    print('stop here')
            if email_contents == True and not('Newsgroups:' in l or 'Path:' in l or 'From:' in l or 'Subject:' in l or 'Organization:' in l or 'Distribution:' in l or 'Date:' in l or 'Message-ID:' in l or 'Xref:' in l or 'Sender:' in l or 'References:' in l):
                if classified == False:
                    print('stop here')
                current_words = tokenizer.tokenize(l.lower())
                current_words = [lemmatizer.lemmatize(i) for i in current_words]
                for word in current_words:
                    if word.lower() in lexicon:
                        index_value = lexicon.index(word.lower())
                        features[index_value] += 1
            if start_tag in l and email_contents == False:
                    email_contents = True
        features = list(features)
        featureset.append([features,the_label])
    return featureset
def generate_features(training_dir,validation_dir,lexicon):
    train_list = os.listdir(training_dir)
    val_list = os.listdir(validation_dir)
    train_features = []
    validation_features = []
    for files in train_list:
        file_path = os.path.join(training_dir, files)
        train_features += sample_handling(file_path,lexicon)
    for files in val_list:
        file_path = os.path.join(validation_dir, files)
        validation_features += sample_handling(file_path,lexicon) 
    random.shuffle(train_features) #to improve the NN training
    random.shuffle(validation_features)
    
    train_features = np.array(train_features)
    validation_features = np.array(validation_features)
    train_x = list(train_features[:,0][:])
    train_y = list(train_features[:,1][:])
    
    test_x = list(validation_features[:,0][:])
    test_y = list(validation_features[:,1][:])
    return train_x,train_y,test_x,test_y
#split_data(rootdir,training_dir,validation_dir)
my_lex = create_dict(training_dir)    #Este va a crear el diccionario que vamos a usar
train_x,train_y,test_x,test_y = generate_features(training_dir,validation_dir,my_lex)
with open('email_set.pickle','wb') as f:
    pickle.dump([train_x,train_y,test_x,test_y],f)
print('redy')