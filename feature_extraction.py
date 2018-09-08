# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 14:47:00 2018

@author: Chris
"""

import spacy
from spacy.symbols import advcl, advmod, amod, appos, hmod, infmod, meta, neg, nmod, nn, npadvmod, num, number, partmod, poss, possessive, prep, quantmod, rcmod
#from spacy.symbols import DT, PRP, PRP$, #IN, BES, MD, CC, WP, WP$
from spacy.symbols import DET, PUNCT, CCONJ, CONJ, SCONJ, PRON, AUX

import nltk
from nltk.corpus import brown

from math import log, exp, inf


#######################
##Linguistic features##
#######################

nlp = spacy.load('en')

modifiers = [advcl, 
             advmod, 
             amod, 
             appos, 
             hmod, 
             infmod, 
             meta, 
             neg, 
             nmod, 
             nn, 
             npadvmod, 
             num, 
             number, 
             partmod, 
             poss, 
             possessive, 
             prep, 
             quantmod, 
             rcmod]

def linguistic_features(sentence_string):
    #Returns an arrady containing num of right-branching nodes,
    #num of left-branching nodes, branching index, num of premodifiers,
    #num of postmodifiers, and modification index
    
    doc = nlp(sentence_string)
    
    right = 0
    np_right = 0
    left = 0
    np_left = 0
    bw_right_count = 0
    np_bw_right_count = 0
    bw_left_count = 0
    np_bw_left_count = 0
    pre = 0
    np_pre = 0
    post = 0
    np_post = 0
    mw_pre = 0
    np_mw_pre = 0
    mw_post = 0
    np_mw_post = 0
    coordination_ballance = 0
    
    #Obtain counts over all tokens
    for token in doc:
        if token.i < token.head.i:
            left += 1
        elif token.i > token.head.i:
            right += 1
        if token.dep in modifiers and token.head.i > token.i:
            pre += 1
            mw_pre += len([child for child in token.subtree])
        elif token.dep in modifiers and token.head.i < token.i:
            post += 1
            mw_post += len([child for child in token.subtree])
        
        in_right = False
        in_left = False
        if token.i < token.head.i:
            in_left = True
        elif token.i > token.head.i:
            in_right = True
        head = token.head
        while head.dep_ != "ROOT" and False in [in_right, in_left]:
            if head.i < head.head.i:
                in_left = True
            elif head.i > head.head.i:
                in_right = True 
            head = head.head
        if in_right:
            bw_right_count += 1
        if in_left:
            bw_left_count += 1
            
        #Coordination balance
        if token.pos_ == "CCONJ":
            consts = [child for child in token.head.children]
            cconj_index = consts.index(token)
            if cconj_index > 0:
                length_difference = abs(
                        len([ch for ch in consts[cconj_index - 1].subtree]) + 
                        1 - 
                        len([ch for ch in consts[cconj_index + 1].subtree]))
            else:
                length_difference = abs(1 - 
                        len([ch for ch in consts[cconj_index + 1].subtree]))
            if length_difference > coordination_ballance:
                coordination_ballance = length_difference
    
    #Obtain counts over noun phrases
    for chunk in doc.noun_chunks:
        for token in chunk:
            if token.i < token.head.i:
                np_left += 1
            elif token.i > token.head.i:
                np_right += 1
            if token.dep in modifiers and token.head.i > token.i:
                np_pre += 1
                np_mw_pre += len([child for child in token.subtree])
            elif token.dep in modifiers and token.head.i < token.i:
                np_post += 1
                np_mw_post += len([child for child in token.subtree])
            
            in_right = False
            in_left = False
            if token.i < token.head.i:
                in_left = True
            elif token.i > token.head.i:
                in_right = True
            head = token.head
            while head.dep_ != "ROOT" and False in [in_right, in_left]:
                if head.i < head.head.i:
                    in_left = True
                elif head.i > head.head.i:
                    in_right = True 
                head = head.head
            if in_right:
                np_bw_right_count += 1
            if in_left:
                np_bw_left_count += 1
    
    return [right, 
            np_right, 
            left, 
            np_left, 
            pre, 
            np_pre, 
            post, 
            np_post, 
            right - left, 
            np_right - np_left, 
            bw_right_count - bw_left_count,
            np_bw_right_count - np_bw_left_count,
            pre - post,
            np_pre - np_post,
            mw_pre - mw_post,
            np_mw_pre - np_mw_post,
            coordination_ballance]
    
    
def function_word_densities(sentence_string):
    
    doc = nlp(sentence_string)
    
    function = 0
    determiners = 0
    pronouns = 0
    prepositions = 0
    punctuations = 0
    auxiliaries = 0
    conjunctions = 0
    wh_pronouns = 0
    
    function_pos = [DET, 
                    PUNCT, 
                    CCONJ, 
                    CONJ, 
                    SCONJ, 
                    PRON, 
                    AUX]
    
    function_tag = ['DT', 
                    'PRP', 
                    'PRP$', 
                    'IN', 
                    'BES', 
                    'MD', 
                    'CC', 
                    'WP', 
                    'WP$']
    
    for token in doc:
        if token.pos in function_pos or token.tag_ in function_tag:
            function += 1
        if token.pos in [DET] or token.tag_ in ["DT"]:
            determiners += 1
        if token.pos in [PRON] or token.tag_ in ['PRP', 'PRP$', 'WP', 'WP$']:
            pronouns += 1
        if token.tag_ in ["IN"]:
            prepositions += 1
        if token.pos in [PUNCT]:
            punctuations += 1
        if token.pos in [AUX] or token.tag_ in ['BES', 'MD']:
            auxiliaries += 1
        if token.pos in [CONJ, CCONJ, SCONJ] or token.tag_ in ['CC']:
            conjunctions += 1
        if token.tag_ in ['WP', 'WP$']:
            wh_pronouns += 1
    
    return [item / len(doc) for item in [function, 
                                        determiners, 
                                        pronouns, 
                                        prepositions, 
                                        punctuations, 
                                        auxiliaries, 
                                        conjunctions, 
                                        wh_pronouns]]


def constituent_sizes(sentence_string):
    
    doc = nlp(sentence_string)
    
    np_lengths = []
    ajp_lengths = []
    pp_lengths = []
    avp_lengths = []
    
    for chunk in doc.noun_chunks:
        np_lengths.append(len(chunk))
        
    for token in doc:
        if token.dep == amod:
            ajp_lengths.append(len([child for child in token.subtree]))
        if token.dep == prep:
            pp_lengths.append(len([child for child in token.subtree]))
        if token.dep in [advcl, advmod, npadvmod]:
            avp_lengths.append(len([child for child in token.subtree]))
    
    for result in [np_lengths, ajp_lengths, pp_lengths, avp_lengths]:
        if len(result) == 0:
            result.append(0)  
    
    return [sum(np_lengths)/len(np_lengths), 
            max(np_lengths), 
            sum(ajp_lengths)/len(ajp_lengths), 
            max(ajp_lengths), 
            sum(pp_lengths)/len(pp_lengths), 
            max(pp_lengths), 
            sum(avp_lengths)/len(avp_lengths), 
            max(avp_lengths), 
            len(doc)]
 
    
##############
##Perplexity##
##############

freq_brown_1gram = nltk.FreqDist(brown.words())
len_brown = len(brown.words())

cfreq_brown_2gram = nltk.ConditionalFreqDist(nltk.bigrams(brown.words()))
cprob_brown_2gram = nltk.ConditionalProbDist(cfreq_brown_2gram, 
                                             nltk.MLEProbDist)

brown_trigrams = nltk.ngrams(brown.words(), 3)
condition_pairs = (((w0, w1), w2) for w0, w1, w2 in brown_trigrams)
cfreq_brown_3gram = nltk.ConditionalFreqDist(condition_pairs)
cprob_brown_3gram = nltk.ConditionalProbDist(cfreq_brown_3gram, 
                                             nltk.MLEProbDist)


def unigram_prob(word):
    return freq_brown_1gram[word] / len_brown

def bigram_prob(word1, word2):
    return cprob_brown_2gram[word1].prob(word2)

def trigram_prob(word1, word2, word3):
    return cprob_brown_3gram[(word1, word2)].prob(word3)

def log_inf(x):
    return log(x) if x > 0 else -inf

def sentence_prob(sentence):
    sentence = sentence.split()
    return (unigram_prob(sentence[0]) * 
           bigram_prob(sentence[0], sentence[1]) * 
           exp(sum(map(log_inf, 
                       [trigram_prob(sentence[i], 
                                     sentence[i+1], 
                                     sentence[i+2]) for i in range(
                                             len(sentence)-2)]))))









