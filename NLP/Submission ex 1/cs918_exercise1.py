# -*- coding: utf-8 -*-

import json
import re
import nltk
nltk.data.path.append('/modules/cs918/nltk_data/')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 
from nltk.util import ngrams
from collections import Counter

wnl = WordNetLemmatizer() 

"""url regexs"""
url_reg2 = r"https?[\S]+" 
url_reg = r"((https?)\:\/\/)?(www\.)?(([a-zA-z0-9-]+)(\.))+((com)|(be)|(ly)|(ca)|(edu)|(gl)|(co\.uk)|(net)|(org(\.uk)?)|(gov(\.uk)?))(\.(\/)?)?((\/[\S]+)+)?"

"""Stores tokens in this list"""
tokens = [] #token list of corpus

"""Counters"""
positive_stories = 0 #number of stories with more positive words
negative_stories = 0 #number of stories with more negative words

positive_count = 0 #number of positive words in corpus
negative_count = 0 #number of negative words in corpus


""" Finds position tag for POS tagging"""
def get_pos_tag(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
""" Open positive words file"""
pr = set(open("positive-words.txt").read().split())


""" Open negative words file"""
nr = set(open("negative-words.txt").read().split())
    
""" Counts positive words """   
def pos_word_count(test_list):
    return sum(i in pr for i in test_list)

""" Counts negative words """
def neg_word_count(test):
    return sum(i in nr for i in test)
    
""" Part A """

print("\n \n ===== Part A =====")

print("\n preprocessing text and calculating positive & negative counts...")

with open("signal-news1.jsonl") as f:
    for line in f: #string form of json object
        tmp = json.loads(line) #each json object loaded as dict
        content = tmp["content"] #fetches only content field
        lower_content = content.lower() #lowercases content
        urlr2 = re.sub(url_reg2, "", lower_content) #removes URLs
        urlr = re.sub(url_reg, "", urlr2) #removes remaining URLs 
        stripped = re.sub(r"([^\s\w]|_)+", "", urlr) #removes all non-alphanumeric characters except spaces
        removed_numb = re.sub(r"\b[0-9]+\b", "", stripped) #removes numbers
        removed_char = re.sub(r"\b\w{1}\b", "", removed_numb) #removes words with 1 character
        split_text = removed_char.split() #converts the text into list of words (tokens)
#        tagged = nltk.pos_tag(split_text)
#        lem_text = [wnl.lemmatize(i[0], get_pos_tag(i[1])) for i in tagged] #lemmatizes the list
        lem_text = [wnl.lemmatize(i) for i in split_text] #lemmatizes the list
        tokens += lem_text # adds the tokens to the token list of corpus
        p_count = pos_word_count(lem_text)
        n_count =  neg_word_count(lem_text)
        if p_count > n_count:
            positive_stories += 1
        elif p_count < n_count:
            negative_stories += 1
        positive_count += p_count
        negative_count += n_count

print("\n preprocessing finished")

""" Part B"""

print("\n \n ===== Part B =====")

types = set(tokens) #finds the types from the tokens of the corpus

print("\n Number of types (vocabulary size):", len(types))        
print("\n Number of tokens:", len(tokens))   

trigrams = ngrams(tokens, 3) #finds trigrams in corpus
cnt = Counter(trigrams) #counts the trigrams
top_trigrams = cnt.most_common(25) #returns top 25 trigrams

print("\n The top 25 trigrams are: \n", top_trigrams)
print("\n Number of positive words in corpus:", positive_count)
print("\n Number of negative words in corpus:", negative_count)
print("\n Number of stories with more positive words: ", positive_stories)
print("\n Number of stories with more negative words: ", negative_stories)

