#!/usr/bin/env python
# coding: utf-8

# In[1]:


###Pip install instructions:
##1. install git software link: https://git-scm.com/download/win
##2. Add the git/bin and git/cmd paths to system paths in environmental variables
##3. Copy the below hastagged text untagged into a Requirements.txt OR Pip install the below detailed packages
##install requirements.txt:

#click==8.0.3
#colorama==0.4.4
#Cython==0.29.23
#gensim==4.1.2
#gsdmm @ git+https://github.com/rwalk/gsdmm.git@4ad1b6b6976743681ee4976b4573463d359214ee
#joblib==1.1.0
#lightgbm==3.3.1
#nltk==3.6.5
#numpy==1.21.4
#pandas==1.3.4
#pyphen==0.11.0
#python-dateutil==2.8.2
#pytz==2021.3
#regex==2021.11.10
#scikit-learn==1.0.1
#scipy==1.7.3
#six==1.16.0
#smart-open==5.2.1
#textfeatures==0.0.2
#textstat==0.7.2
#threadpoolctl==3.0.0
#tqdm==4.62.3

##Trying to pip install the gsdmm package should be done like this (untagged):
#pip install git+https://github.com/rwalk/gsdmm.git


# In[2]:


###Importing and downloading packages

import pandas as pd
import textfeatures as tf
import textstat as ts
import nltk
import numpy as np
import gsdmm #
import gensim
import json
import lightgbm as lgb


from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict
from gensim import corpora
from gsdmm import MovieGroupProcess
from nltk.tokenize import word_tokenize
from collections import Counter
from pathlib import Path
from pandas.io.json import json_normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

nltk.download('wordnet')
nltk.download('stopwords')


# In[3]:


###Function for importing external data from Google Spreadsheets

#Source1 of table: http://cic.tju.edu.cn/faculty/zhileiliu/doc/COREComputerScienceConferenceRankings.html

#Input of the below function is the URL id of the google drive spreadsheet
def import_data_from_spreadsheet(my_sheet_id):
    sheet_id = my_sheet_id
    r = "https://docs.google.com/spreadsheets/export?id={}&exportFormat=csv".format(sheet_id)
    df = pd.read_csv(r, engine="python", encoding="utf8", error_bad_lines=False)
    df = df.rename({'venue': 'venue_cleaned'}, axis=1)
    #Function returns a pandas dataframe
    return df


# In[4]:


###Function for adding the external author metrics to the dataframe
def add_metrics (df, author_metrics): 
    citations = []
    citations_since_2016 = []
    h_index = []
    h_index_since_2016 = []
    i_index = []
    i_index_since_2016 = []
    for i in range(len (df)):
        citations.append(author_metrics["Citations"][i])
        citations_since_2016.append(author_metrics["Citations Since 2016"][i])
        h_index.append(author_metrics["h-index"][i])
        h_index_since_2016.append(author_metrics["h-index Since 2016"][i])
        i_index.append(author_metrics["i10-index"][i])
        i_index_since_2016.append(author_metrics["i10-index"][i])
    df["total_citations"] = citations
    df["citations_since_2016"] = citations_since_2016
    df["h_index"] = h_index
    df["h_index_since_2016"] = h_index_since_2016
    df["i_index"] = i_index
    df[" i_index_since_2016 "] = i_index_since_2016
    return df


# In[5]:


###Function for extracting basic text features from variables
def desc_text_features(df, text_features):
    ts.set_lang('en_US')
    #df - is a dataframe
    #text_features - list with names of text features that we would like to get features from
    
    df['topics_str'] = df.topics.apply(lambda x: ', '.join([str(i) for i in x]))
    
    for i in text_features:
        df[i].fillna('', inplace = True)
    
    def pct_unique_words(text):
        new_text = text.replace(',', ' ').replace('.', ' ').replace('?', ' ')          .replace('!', ' ').replace('...', ' ').replace('(', ' ').replace(')', ' ').split()
        new_text_list = text.split()
        if len(new_text_list) > 0:
            return len(set(new_text_list))/len(new_text_list) 
        else:
            return len(set(new_text_list))/1
        return
    
    def quasi_sentence_count(text):
        count = 0
        for i in text:
            if i == '.':
                count += 1
        return count
    
    def comma_count(text):
        count = 0
        for i in text:
            if i == ',':
                count += 1
        return count
    
    def question_count(text):
        count = 0
        for i in text:
            if i == '?':
                count += 1
        return count
    
    def capitals_count(text):
        count = 0
        for i in text:
            if i in 'QWERTYUIOPASDFGHJKLZXCVBNM':
                count += 1
        return count
    
            #count words
    tf.word_count(df, 'title', 'title_word_count')
            #count characters
    tf.char_count(df, 'title', 'title_char_count')
            #count stopwords
    tf.stopwords_count(df, 'title', 'title_stopwords_count')
            #count capitals
    df['title_capitals_count'] = df["title"].map(lambda x: capitals_count(x))
            #compute readability "flesch_reading_ease" score from 'textstat' library 
    tf.word_count(df, 'title', 'title_word_count')
    df['abstract_readability'] = df['abstract'].map(lambda x: ts.flesch_reading_ease(x))
            #compute readability "automated" score from 'textstat' library 
    df['abstract_auto_readability'] = df["abstract"].map(lambda x: ts.automated_readability_index(x))
            
    df['log_abstract_pct_unique_words'] = df["abstract"].map(lambda x: np.log1p(pct_unique_words(x)*100))
    
    df['log_topics_comma_count'] = df["topics_str"].map(lambda x: np.log1p(comma_count(x)))

    tf.word_count(df, "topics_str", 'topics_word_count')
    #following code adopted from https://stackoverflow.com/a/54389030
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    # log transform numeric columns
    for c in [c for c in df.columns if df[c].dtype in numerics and 'log_' not in c              and 'readability' not in c]:
        df['log_{}'.format(c)] = np.log1p(df[c])
    
    return df


# In[6]:


###Function for cleaning venues in order to create dummy variables from most frequent ones
def clean_venue(df, list_of_dictionaries):
    venue_cleaned = []
    
    for dictionary in list_of_dictionaries:
        venue = dictionary["venue"]
        venue = venue.lower()
        
        if "@" in venue:    ## remove "@" - @s are used for workshops within conferences - if we want to make nr. of unique venus lower, this is crucial 
            venue = venue[:venue.find("@")]
            
        if " 19" in venue:  ## finding years 19xx - removing those to unify them with conference
            venue = venue[:venue.find(" 19")]
            
        if " 20" in venue:  ## finding years 20xx - removing those to unify them with conference
            venue = venue[:venue.find(" 20")]
            
        if "/" in venue:  ## same as above, but we take IJCNLP, we 
            venue = venue[venue.find("/")+1:]
            
        if "*" in venue:    ## removing * from *SEMEVAL and *SEM
            venue = venue[venue.find("*")+1:]  
            
        if " " in venue:       ## searching for venues with longer string - finding abbrv.
            venue = venue.split(" ")  
            for word in venue:     
                up = 0      ## HELP variable - how many uppercasses are in word
                for letter in word:  
                    if letter.isupper():
                        up += 1  ## for each uppercase in word +1
                if up > 1:   ## if there are more uppercases -> abbreviation
                    venue = word    
            if type(venue) == list:   ##if we are left with list, no abbreviation found
                venue = " ".join(venue)
                
        if "-" in venue:    ## remove "-" - we are not sure about this, but at least based on distribution, this seems to be correct way 
            venue = venue[:venue.find("-")]
        venue = venue.upper()
        venue_cleaned.append(venue)
        
    ## IDENTFIY THE MOST FREQUENT VALUES TO CREATE DUMMIES
    occurence_count = Counter(venue_cleaned)  ## counter of items in list
    venue_dummies = []
    occurence_count = sorted(occurence_count.items(),key=lambda x:-x[1]) ## change to items, sorting (ML)
    
    for value in range(len(occurence_count)): ## unpacking counter
        x,y = occurence_count[value] 
        occurence_count[value] = x ## returning ordered list of venues
    df["venue_cleaned"] = venue_cleaned
    
    for venue in venue_cleaned:
        if len(venue) == 0:   ## empty venues changed "empty" -> up to our discussion
            venue_dummies.append("empty")
        elif venue in list(occurence_count)[:7]: ## venues which occured in top 6 kept (empty solved earlier)
            venue_dummies.append(venue)
        else:   ## other replaced with "other"
            venue_dummies.append("other_venue")
    df["venue_cleaned2"] = venue_dummies
    
    ## DUMMY VARIABLES CREATION
    dummy_venue = pd.get_dummies(df["venue_cleaned2"])
    df = pd.concat([df, dummy_venue], axis = 1)
    return df


# In[7]:


####CREATE A NEW COLUMN WITH AUTHOR COUNTS, ADD THEM AND A LOG TRANSFORMED VERSION TO DF
def count_number_of_authors (list_of_dictionaries, df): #(???)
    number_of_authors= []
    
    for dictionary in list_of_dictionaries:
        count=0
        for author in dictionary["authors"]:
            count+= 1   #count number of authors out of the list of dictionaries
        number_of_authors.append(count) #add the counts to a new list
        
    df["number_of_authors"] = number_of_authors #add the list to the dataframe
    df["number_of_authors_log"] = np.log1p(df["number_of_authors"]+1)
    
    return df #return df with two new columns (number of authors and log transformed version of number of authors)


# In[8]:


###EXTRACT DOI NUMBER AND ADD DUMMY VARIABLE FOR DIFFERENT VERSIONS OF DOI TO DF
def extract_doi_number (list_of_dictionaries, df):
    doi_cleaned = []
    
    for dictionary in list_of_dictionaries: 
        full_doi = dictionary["doi"]
        start = full_doi.find(".")
        end = full_doi.find("/")
        doi_cleaned.append(full_doi[start+1:end])
        
    occurence_count = Counter(doi_cleaned)  ## counter of items in list
    doi_dummies = []
    occurence_count = sorted(occurence_count.items(),key=lambda x:-x[1]) ## change to items, sorting (ML)
    
    for value in range(len(occurence_count)): ## unpacking counter
        x,y = occurence_count[value] 
        occurence_count[value] = x 
        
    for doi in doi_cleaned:
        if len(doi) == 0:   
            doi__dummies.append("no doi")
        elif doi in list(occurence_count)[:3]: 
            doi_dummies.append(doi)
        else:   
            doi_dummies.append("other_doi")
            
    df["doi_cleaned"] = doi_dummies
    dummy_doi = pd.get_dummies(df["doi_cleaned"])
    df = pd.concat([df, dummy_doi], axis = 1)
    
    return df
#Test data has same DOI numbers


# In[9]:


#get dummy variable for is_open_acess
def dummy_open_acess(df): 
    dummy_is_open_access = pd.get_dummies(df["is_open_access"])
    df = pd.concat([df, dummy_is_open_access], axis = 1)
    df.rename(columns = {True : "is open access: True", False: "is open access: False"}, inplace = True)
    return df


# In[10]:


###Log transform citations and references
def add_log_transformed_variable_to_df(df, column_name):
    df[column_name + "_log"] = np.log1p(df[column_name])
    return df


# In[11]:


### Function for applying lda topic modelling on abstracts

#Source: https://notebook.community/gojomo/gensim/docs/src/auto_examples/tutorials/run_lda


#Input: pandas read in json files
def lda_preprocessing(train_data, test_data, num_topics, min_count_grams, chunksize = 2000, passes = 20, iterations = 400):

    #Concatenating the train and test dataset to run LDA on both in order to let us test our model later
    train_test = pd.concat([train_data[['doi', 'abstract']], test_data[['doi', 'abstract']]])
    train_test = train_test.drop_duplicates(subset = 'doi')

    #Create indicator variable which shows whether topic variable is empty or not
    train_test['abstract'].fillna("",inplace=True)
    train_test['abstract_empty'] = np.where(train_test['abstract'] == '', 1, 0)

    #Get new data set with only papers with topics
    non_empty_abstract_df  = train_test[train_test['abstract_empty'] < 1]

    #Create a docs object in order to feed data into learning algorithm
    docs = non_empty_abstract_df["abstract"].astype("string").tolist()

    # Split the documents into tokens.
    tokenizer = RegexpTokenizer(r'\w+')
    for index in range(len(docs)):
        docs[index] = str(docs[index]).lower()  # Convert to lowercase.
        docs[index] = tokenizer.tokenize(docs[index])  # Split into words.

    # Remove numbers, but not words that contain numbers.                           
    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

    # Remove words that are only two character.                                     
    docs = [[token for token in doc if len(token) > 2] for doc in docs]

    #Remove stopwords
    stop_words = set(stopwords.words('english'))
    docs = [[token for token in doc if not token in stop_words] for doc in docs]

    # Lemmatize the documents.
    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

    # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
    bigram = Phrases(docs, min_count=min_count_grams)
    for index in range(len(docs)):
        for token in bigram[docs[index]]:
            if '_' in token:
              # Token is a bigram, add to document.
              docs[index].append(token)

    # Create a dictionary representation of the documents.
    dictionary = Dictionary(docs)

    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=20, no_above=0.5)

    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    # Set training parameters.
    num_topics = num_topics
    chunksize = chunksize
    passes = passes
    iterations = iterations
    eval_every = None #Be careful -> takes a lot of time!

    # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    #Initialize LDA model
    model = LdaModel(
      corpus=corpus,
      id2word=id2word,
      chunksize=chunksize,
      alpha='auto',
      eta='auto',
      iterations=iterations,
      num_topics=num_topics,
      passes=passes,
      eval_every=eval_every
    )

    #Source: https://groups.google.com/g/gensim/c/CJZc7KN60JE/m/9Xs8C0YMgq4J (deleted comment)
    #Create a document and topic matrix
    def get_doc_topic(corpus, model):
        doc_topic = list()
        for doc in corpus:
            doc_topic.append(model.__getitem__(doc, eps=0))
        return doc_topic
    doc_topic = get_doc_topic(corpus, model)

    #Creating pandas DF from document-topic matrix of gensim LDA algorithm
    list_of_lists = []
    for index, document_topics in enumerate(doc_topic):
        inner_list = []
        inner_list.append(index+1)
        for i in document_topics:
            inner_list.append(i[1])
        list_of_lists.append(inner_list)
    document_topic_df = pd.DataFrame(list_of_lists)
    document_topic_df.columns = document_topic_df.columns.map(str)

    # #Merge (join) the pandas Dataframes on the key column which contains the identifier of the documents
    non_empty_abstract_df.reset_index(drop=True, inplace=True)
    document_topic_df.reset_index(drop=True, inplace=True)
    result = pd.concat([non_empty_abstract_df, document_topic_df], axis=1)
    topic_header_num = num_topics + 2
    
    for i in range(len(result.columns.values.tolist()[1:topic_header_num])):
        result = result.rename(columns={'{}'.format(i) : 'Topic_abstract_{}'.format(i)})

    #Merge the lda_df with the original data sets
    train_data = pd.merge(train_data, result, how="left", on="doi")
    test_data = pd.merge(test_data, result, how="left", on="doi")

    #Fill the missing values at the topics with 0
    train_data["abstract_empty"].fillna(1,inplace=True)
    for i in range(1,topic_header_num-1):
        train_data['Topic_abstract_{}'.format(i)].fillna(0,inplace=True)
    test_data["abstract_empty"].fillna(1,inplace=True)
    
    for i in range(1,topic_header_num-1):
        test_data['Topic_abstract_{}'.format(i)].fillna(0,inplace=True)

    #Clean up the unecessary columns and rename which is needed in both original sets
    train_data = train_data.drop(columns='Topic_abstract_0')
    train_data = train_data.drop(columns='abstract_y')
    train_data = train_data.drop(columns='abstract_empty')
    train_data.rename(columns={'abstract_x': 'abstract'}, inplace=True)
    test_data = test_data.drop(columns='Topic_abstract_0')
    test_data = test_data.drop(columns='abstract_y')
    test_data = test_data.drop(columns='abstract_empty')
    test_data.rename(columns={'abstract_x': 'abstract'}, inplace=True)

    #Output pandas dataframe objects
    return train_data, test_data


# In[12]:


###Topic modelling with short text gsdmm on topics

def gsdmm_topics_on_topics(train_json, test_json, n_topics=20, a=0.1, b=0.1, n_iterations=20):

    #Combine train and test sets
    train_test = pd.concat([train_json[['doi', 'topics']], test_json[['doi', 'topics']]])
    #Drop duplicates
    train_test = train_test.drop_duplicates(subset = 'doi')
    #Convert topics variable from list to string
    train_test['topics_str'] = train_test.topics.apply(lambda x: ', '.join([str(i) for i in x]))
    #Create indicator variable which shows whether topic variable is empty or not
    train_test['topic_topics_empty'] = np.where(train_test['topics_str'] == '', 1, 0)

    #Get new data set with only papers with topics
    non_empty_topics_df = train_test[train_test['topic_topics_empty'] < 1]

    #TOPIC MODELLING for short texts. Code was adopted and adapted from this source:
    #https://towardsdatascience.com/short-text-topic-modelling-lda-vs-gsdmm-20f1db742e14

    docs = non_empty_topics_df.topics_str.to_numpy()
    docs = [d.lower().split(', ') for d in docs]

    # create dictionary of all words in all documents
    dictionary = corpora.Dictionary(docs)
    
    # filter extreme cases out of dictionary
    dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=100000)
    
    
    # create variable containing length of dictionary/vocab
    vocab_length = len(dictionary)

    # create BOW dictionary
    bow_corpus = [dictionary.doc2bow(doc) for doc in docs]
    
    # initialize GSDMM
    gsdmm = MovieGroupProcess(K=n_topics, alpha=a, beta=b, n_iters=n_iterations)
    # fit GSDMM model
    y = gsdmm.fit(docs, vocab_length)
    
    #Create list of topic probabilities for each document
    list_of_topic_prob = []
    for i in docs:
        list_of_topic_prob.append(gsdmm.score(i))
        
    #Create new variable with list of topic belonging probabilities 
    non_empty_topics_df = non_empty_topics_df.assign(topic_topics = list_of_topic_prob)
    
    #Merge topic probabilities to train_test data
    train_test = pd.merge(train_test, non_empty_topics_df[['doi', 'topic_topics']], how = 'left', on='doi')
    
    #Create list with 0. for papers with no topics/keywords (missing values)
    null_list = []
    for i in range(0, n_topics):
        null_list.append(0.)
        
    #Assign list of 0. to topics prob variable for papers with no topics/keywords (missing values)
    train_test['topic_topics'] = [ null_list if topic_topics is np.NaN                                   else topic_topics for topic_topics                                   in train_test['topic_topics'] ]
    
    #Create column names for topics' probabilities variables
    topics_var_list = []
    for i in range(0, n_topics):
        topics_var_list.append('topic_topics_' + str(i))
    
    #https://datascienceparichay.com/article/split-pandas-column-of-lists-into-multiple-columns/
    #new df from the column of lists
    split_df = pd.DataFrame(train_test['topic_topics'].tolist(), columns=topics_var_list)
    
    #concat train_test and split_df
    train_test = pd.concat([train_test, split_df], axis=1)
    
    #Merging topic probabilities back to train and test data
    train_df = pd.merge(train_json, train_test[['doi', 'topic_topics']+topics_var_list],                          how = 'left', on='doi')
    
    test_df = pd.merge(test_json, train_test[['doi', 'topic_topics']+topics_var_list],                          how = 'left', on='doi')

      
    return train_df, test_df #(renamed from _json!)


# In[13]:


###Topic modelling with short text gsdmm on titles

#Source: https://datascienceparichay.com/article/split-pandas-column-of-lists-into-multiple-columns/

def gsdmm_topics_on_titles(train_json, test_json, n_topics=20, a=0.1, b=0.1, n_iterations=20):

    #Combine train and test sets
    train_test = pd.concat([train_json[['doi', 'title']], test_json[['doi', 'title']]])

    #Drop duplicates
    train_test = train_test.drop_duplicates(subset = 'doi')
    train_test.reset_index()
    docs = train_test["title"].astype("string").tolist()

    # Split the documents into tokens.
    tokenizer = RegexpTokenizer(r'\w+')
    for index in range(len(docs)):
        docs[index] = str(docs[index]).lower()  # Convert to lowercase.
        docs[index] = tokenizer.tokenize(docs[index])  # Split into words.

    # Remove numbers, but not words that contain numbers.                           
    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

    # Remove words that are only one character.                                     
    docs = [[token for token in doc if len(token) > 2] for doc in docs]

    # Remove stopwords 
    stop_words = set(stopwords.words('english'))
    docs = [[token for token in doc if not token in stop_words] for doc in docs]

    # Lemmatize the documents.
    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

    # Add bigrams and trigrams to docs (only ones that appear 20 times or more).     
    bigram = Phrases(docs, min_count=20)
    for index in range(len(docs)):
        for token in bigram[docs[index]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[index].append(token)

    # create dictionary of all words in all documents
    dictionary = corpora.Dictionary(docs)

    # filter extreme cases out of dictionary
    dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=10000)


    # create variable containing length of dictionary/vocab
    vocab_length = len(dictionary)

    # create BOW dictionary
    bow_corpus = [dictionary.doc2bow(doc) for doc in docs]

    # initialize GSDMM
    gsdmm = MovieGroupProcess(K=n_topics, alpha=a, beta=b, n_iters=n_iterations)
    
    # fit GSDMM model
    y = gsdmm.fit(docs, vocab_length)

    #Create list of topic probabilities for each document
    list_of_topic_prob = []
    for i in docs:
        list_of_topic_prob.append(gsdmm.score(i))

    #Create new variable with list of topic belonging probabilities 
    train_test = train_test.assign(title_topics = list_of_topic_prob)
    train_test = train_test.reset_index()

    topics_var_list = []
    for i in range(0, n_topics):
        topics_var_list.append('title_topics' + str(i))   

    # new df from the column of lists
    split_df = pd.DataFrame(train_test['title_topics'].tolist(), columns=topics_var_list)

    # concat train_test and split_df
    train_test = pd.concat([train_test, split_df], axis=1)

    #Merging topic probabilities back to train and test data
    train_df = pd.merge(train_json, train_test[['doi', 'title_topics']+topics_var_list],                           how = 'left', on='doi')

    test_df = pd.merge(test_json, train_test[['doi', 'title_topics']+topics_var_list],                           how = 'left', on='doi')  

    return train_df, test_df #Renamed from _json!

# In[15]:


###IMPORTING DATA

# path to file
p1 = Path('train-1.json')
p2 = Path('test.json')

# read the file in and load using the json module
with p1.open('r', encoding='utf-8') as f:
    train = json.loads(f.read())
with p2.open('r', encoding='utf-8') as f:
    test = json.loads(f.read())

# create a dataframe
train_df = json_normalize(train)
test_df = json_normalize(test)

f = open("train-1.json")
train_dict = json.load(f) #load data into a list of dictionaries

f_test = open("test.json")
test_dict = json.load(f_test)

###Generate variables from basic descriptive features of title and abstract
train_df = desc_text_features(train_df, ['title', 'abstract'])
test_df = desc_text_features(test_df, ['title', 'abstract'])

#Run the below function to get the dataframe with the extra author metrics information
author_metrics = import_data_from_spreadsheet("DELETED_SOURCE") #Put here the ID of your googlesheet!

author_metrics_train = author_metrics.iloc[:9658]
author_metrics_train = author_metrics_train.reset_index()

author_metrics_test = author_metrics.iloc[9658:]
author_metrics_test = author_metrics_test.reset_index()


# In[16]:


###Rund Tfidf Vectorizer on title and abstract

train_df["abstract"].fillna('', inplace = True)
test_df["abstract"].fillna('', inplace = True)

vec1 = TfidfVectorizer(stop_words="english", analyzer='word', ngram_range=(1, 2), max_df=0.95, min_df=5, max_features=50)
vec2 = TfidfVectorizer(stop_words="english", analyzer='word', ngram_range=(1, 2), max_df=0.95, min_df=5, max_features=50)

train_counts1 = vec1.fit_transform(train_df.title)
test_counts1 = vec1.transform(test_df.title)

train_counts2 = vec2.fit_transform(train_df.abstract)
test_counts2 = vec2.transform(test_df.abstract)

train_counts1_df = pd.DataFrame(train_counts1.toarray(), columns=vec1.get_feature_names(), index= train_df.index)
test_counts1_df = pd.DataFrame(test_counts1.toarray(), columns=vec1.get_feature_names(), index= test_df.index)

for i in range(len(train_counts1_df.columns)):
    train_counts1_df = train_counts1_df.rename(columns={'{}'.format(train_counts1_df.columns[i]) : 'tfidf_title_{}'.format(train_counts1_df.columns[i])})

for i in range(len(test_counts1_df.columns)):
    test_counts1_df = test_counts1_df.rename(columns={'{}'.format(test_counts1_df.columns[i]) : 'tfidf_title_{}'.format(test_counts1_df.columns[i])})

train_df = train_df.join(train_counts1_df)
test_df = test_df.join(test_counts1_df)

train_counts2_df = pd.DataFrame(train_counts2.toarray(), columns=vec2.get_feature_names(), index= train_df.index)
test_counts2_df = pd.DataFrame(test_counts2.toarray(), columns=vec2.get_feature_names(), index= test_df.index)

for i in range(len(train_counts2_df.columns)):
    train_counts2_df = train_counts2_df.rename(columns={'{}'.format(train_counts2_df.columns[i]) : 'tfidf_abstract_{}'.format(train_counts2_df.columns[i])})

for i in range(len(test_counts2_df.columns)):
    test_counts2_df = test_counts2_df.rename(columns={'{}'.format(test_counts2_df.columns[i]) : 'tfidf_abstract_{}'.format(test_counts2_df.columns[i])})

train_df = train_df.join(train_counts2_df)
test_df = test_df.join(test_counts2_df)

###Adding author metrics to both train and test dataset for prediction
train_df = add_metrics (train_df, author_metrics_train)
test_df = add_metrics (test_df, author_metrics_test)

###Cleaning and dummy encoding venues
train_df = clean_venue(train_df, train_dict)
test_df = clean_venue(test_df, test_dict)

###Run the below function to get the dataframe with the extra conference ranking information

#Source: https://www.askpython.com/python/examples/impute-missing-data-values; http://cic.tju.edu.cn/faculty/zhileiliu/doc/COREComputerScienceConferenceRankings.html

conference_ranking_df = import_data_from_spreadsheet("DELETED_SOURCE") #Put here the ID of your googlesheet!
conference_ranking_df= conference_ranking_df[["venue_cleaned","Rank"]]

train_df = pd.merge(train_df, conference_ranking_df, how="left", on="venue_cleaned")
test_df = pd.merge(test_df, conference_ranking_df, how="left", on="venue_cleaned")


train_df["Rank"] = train_df["Rank"].fillna(value="NA")
test_df["Rank"] = test_df["Rank"].fillna(value="NA")

train_df = pd.concat([train_df.drop(columns=["Rank"]), pd.get_dummies(train_df["Rank"])], 1)
test_df = pd.concat([test_df.drop(columns=["Rank"]), pd.get_dummies(test_df["Rank"])], 1)

###We need to drop B and C since they do not occur in test data and occur only several times in train
train_df.drop(['B', 'C'], axis = 1, inplace = True)

###Counting number of authors
train_df = count_number_of_authors(train_dict, train_df)
test_df = count_number_of_authors(test_dict, test_df) #(I ADDED test_dict in the first cell - idk if it's ok)

###Extracting information from doi links
train_df = extract_doi_number (train_dict, train_df)
test_df = extract_doi_number (test_dict, test_df)

###Creating dummy variables from open access documents
train_df = dummy_open_acess(train_df)
test_df = dummy_open_acess(test_df)

###Handling missing data after integration

#code from https://www.askpython.com/python/examples/impute-missing-data-values
missing_col = ['year'] #impute missing data for year with the median for year
for i in missing_col:
    train_df.loc[train_df.loc[:,i].isnull(),i]=train_df.loc[:,i].median()
missing_col = ['year']

for i in missing_col:
    test_df.loc[test_df.loc[:,i].isnull(),i]=test_df.loc[:,i].median()

train_df['year'] = train_df['year'].astype(int) # convert year to int, its a bit strange to have a float for year
test_df['year'] = test_df['year'].astype(int)

###Applying log transformation
train_df = add_log_transformed_variable_to_df (train_df, "citations")
train_df = add_log_transformed_variable_to_df (train_df, "references")
test_df = add_log_transformed_variable_to_df (test_df, "references")

###Applying lda on abstract 
train_df, test_df = lda_preprocessing(train_df, test_df, num_topics = 9, min_count_grams = 30)

###Applying short text topic modelling on topics
train_df, test_df = gsdmm_topics_on_topics(train_df, test_df)

###Applying short text topic modelling on titles
train_df, test_df = gsdmm_topics_on_titles(train_df, test_df)

###Renaming variables for clarity's sake

train_df = train_df.rename(columns={'A+':"A_plus", '1162':'doi_1162',                                    '18653':'doi_18653', '3115':'doi_3115', 
                                   'is open access: True': 'is_open_access'})
test_df = test_df.rename(columns={'A+':"A_plus", '1162':'doi_1162',                                    '18653':'doi_18653', '3115':'doi_3115', 
                                   'is open access: True': 'is_open_access'})


# In[17]:


###Removing string variables before modelling

train_df = train_df.drop(['doi', 'title', 'abstract', 'authors', 'venue', 'references',
       'topics', 'is_open_access', 'fields_of_study', 'citations',
       'topics_str', 'doi_cleaned', 'topic_topics', 'doi_cleaned', 
       'log_year', 'is open access: False','venue_cleaned', 'log_references', 'log_citations', 
       "number_of_authors", "title_char_count", 'venue_cleaned2',
       'log_topics_comma_count', 'title_topics'], axis = 1)

test_df = test_df.drop(['doi', 'title', 'abstract', 'authors', 'venue', 'references',
       'topics', 'is_open_access', 'fields_of_study', 
       'topics_str', 'doi_cleaned', 'topic_topics', 'doi_cleaned', 
       'log_year', 'is open access: False','venue_cleaned', 'log_references', 
       "number_of_authors", "title_char_count",'venue_cleaned2', \
              'log_topics_comma_count', 'title_topics'], axis = 1)


# In[18]:


###Applying Light Gradient Boosting Machine

#Selecting and Preparing data
target = train_df['citations_log']
features = train_df.drop(columns = ['citations_log'])
test_features = test_df.copy()

#Scaling the data
scaler = StandardScaler()

#Transform data
features = scaler.fit_transform(features)
test_features = scaler.transform(test_features)

#Setting best fine tuned model
LGBM = lgb.LGBMRegressor(objective = 'regression', 
                         boosting_type = 'gbdt',
                         num_leaves = 120,
                         learning_rate = 0.010345690405573949, 
                         min_child_samples = 30, 
                         reg_alpha = 1.0,
                         reg_lambda = 0.4444444444444444,
                         colsample_bytree = 0.7333333333333333,
                         subsample = 0.8333333333333333,
                         n_estimators = 965, 
                         random_state = 532)


# In[19]:


#Fitting the final model on joined training and validation data
LGBM = LGBM.fit(features, target)

#Predicting the test data
y_pred = LGBM.predict(test_features)

#Preparing predicted citation scores for extraction
predicted_counts = np.expm1(y_pred)
test_df_2 = json_normalize(test)
pred_df = pd.concat([test_df_2['doi'], pd.DataFrame(predicted_counts)], axis=1)
pred_df.columns = ['doi', 'citations']
pred_dict = pred_df.to_dict('records')

#Extracting predictions into a json file
import json
with open('predicted.json', 'w') as fout:
    json.dump(pred_dict, fout)

    
    
###APPENDIX:
    
    
##FINE TUNING LIGHTGBM
#The code adopted from
#https://www.kaggle.com/willkoehrsen/intro-to-model-tuning-grid-and-random-search

#target = train_df['citations_log']
#features = train_df.drop(columns = ['citations_log'])
#test_features = test_df
#train_set = lgb.Dataset(data = features, label = target)
#test_set = lgb.Dataset(data = test_features)

#def objective(hyperparameters, iteration):
 #   """Objective function for grid and random search. Returns
  #     the cross validation score from a set of hyperparameters."""
    
    # Number of estimators will be found using early stopping
   # if 'n_estimators' in hyperparameters.keys():
    #    del hyperparameters['n_estimators']
    
     # Perform n_folds cross validation
    #cv_results = lgb.cv(hyperparameters, train_set, num_boost_round = 10000, nfold = 5, 
     #                   early_stopping_rounds = 100, metrics = 'mse', seed = 51, 
      #                  stratified = False)
    
    # results to retun
    #score = cv_results['l2-mean'][-1]
    #estimators = len(cv_results['l2-mean'])
    #hyperparameters['n_estimators'] = estimators 
    
    #return [score, hyperparameters, iteration]

# Hyperparameter grid
#param_grid = {
    #'objective':['regression'],
    #'boosting_type': ['gbdt'],
    #'n_estimators': list([1000, 3000, 5000, 10000]),
   # 'num_leaves': list([20, 30, 50, 90, 120, 150]),
    #'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 20)),
   # 'min_child_samples': [10, 15, 20, 25, 30, 40, 50],
   # 'reg_alpha': list(np.linspace(0, 1, 10)),
   # 'reg_lambda': list(np.linspace(0, 1, 10)),
   # 'colsample_bytree': list(np.linspace(0.6, 1, 10)),
   # 'subsample': list(np.linspace(0.5, 1, 10)),
#}

#import numpy as np
#import random

#random.seed()

# Randomly sample from dictionary
#random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
# Deal with subsample ratio
#random_params['subsample'] = 1.0 if random_params['boosting_type'] == 'goss' else random_params['subsample']

#random_params

#https://www.kaggle.com/willkoehrsen/intro-to-model-tuning-grid-and-random-search

#def random_search(param_grid, max_evals = 100):
   # """Random search for hyperparameter optimization"""
    
    # Dataframe for results
   # results = pd.DataFrame(columns = ['score', 'params', 'iteration'],
    #                              index = list(range(max_evals)))
   # 
    # Keep searching until reach max evaluations
   # for i in range(max_evals):
        
        # Choose random hyperparameters
    #    hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
     #   hyperparameters['subsample'] = 1.0 if hyperparameters['boosting_type'] == 'goss' else hyperparameters['subsample']

        # Evaluate randomly selected hyperparameters
      #  eval_results = objective(hyperparameters, i)
        
      #  results.loc[i, :] = eval_results
    
    # Sort with best score on top
   # results.sort_values('score', ascending = False, inplace = True)
   # results.reset_index(inplace = True)
   # return results 

#https://www.kaggle.com/willkoehrsen/intro-to-model-tuning-grid-and-random-search

#TABLE WITH RESULTS OF FINE TUNING
#https://www.kaggle.com/willkoehrsen/intro-to-model-tuning-grid-and-random-search

#random_results = random_search(param_grid)

#print('The best validation MSE was {:.5f}'.format(random_results.loc[99, 'score']))
#print('\nThe best hyperparameters were:')

#import pprint
#pprint.pprint(random_results.loc[99, 'params'])

##Lazypredict code workflow

#Source: https://github.com/shankarpandala/lazypredict
#!pip install lazypredict

#from lazypredict.Supervised import LazyRegressor
#from sklearn import datasets
#from sklearn.utils import shuffle
#import numpy as np

#You should load the data here as a (from pandas DF to) list of lists (rows) (but before that filter out the X and y columns into separate objects)
#y = train_df["citations_log"]
#X = train_df.drop("citations_log", axis=1)
#X, y = shuffle(X, y, random_state=13)

#X = X.astype(np.float32)

#Scaling the data
#from sklearn.preprocessing import StandardScaler
# define standard scaler
#scaler = StandardScaler()
# transform data
#X = scaler.fit_transform(X)

#offset = int(X.shape[0] * 0.9)

#Splitting the data into train test sets
#X_train, y_train = X[:offset], y[:offset]
#X_test, y_test = X[offset:], y[offset:]

#Initializing and running the lazyregressor
#reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
#models, predictions = reg.fit(X_train, X_test, y_train, y_test)

#print(models)
