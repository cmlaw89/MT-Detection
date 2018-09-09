# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 17:07:47 2018

@author: Chris
"""

import string
import time
import pickle

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import word_tokenize
from nltk import pos_tag
from nltk import FreqDist, bigrams, trigrams, MLEProbDist, ConditionalFreqDist, ConditionalProbDist, ConditionalProbDistI, LaplaceProbDist


import spacy
from spacy.symbols import advcl, advmod, amod, appos, hmod, infmod, meta, neg, nmod, nn, npadvmod, num, number, partmod, poss, possessive, prep, quantmod, rcmod
from spacy.symbols import DET, PUNCT, CCONJ, CONJ, SCONJ, PRON, AUX



from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split as tts

def timeit(func):
    """
    Simple timing decorator
    """
    def wrapper(*args, **kwargs):
        start  = time.time()
        result = func(*args, **kwargs)
        delta  = time.time() - start
        return result, delta
    return wrapper

def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg

class NLTKPreprocessor(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        
        self.nlp = spacy.load('en')
        self.tri_cprob_dist = ConditionalProbDistI()
        self.modifiers = [advcl, 
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
        self.function_pos = [DET, 
                             PUNCT, 
                             CCONJ, 
                             CONJ, 
                             SCONJ, 
                             PRON, 
                             AUX]
        self.function_tag = ['DT', 
                            'PRP', 
                            'PRP$', 
                            'IN', 
                            'BES', 
                            'MD', 
                            'CC', 
                            'WP', 
                            'WP$']
        
        
        
    def fit(self, X, y=None):
        return self
    
    def inverse_transform(self, X):
        return self
    
    def transform(self, X):
        return [self.vectorize(sent) for sent in X]
        
    def vectorize(self, sentence):
        linguistic = self.linguistic_features(sentence)
        function_word_densities = self.function_word_densities(sentence)
        constituent_sizes = self.constituent_sizes(sentence)
        perplexity = self.perplexity(sentence)
        
        return (linguistic + 
                function_word_densities + 
                constituent_sizes +
                perplexity)
    
    def build_uni_freq_dist(self, sentence_list):
        """
        Returns a frequency distibution of words in a list of sentences
        """
        
        words = [word for sublist in 
                 [word_tokenize(sent.lower()) for sent in sentence_list] 
                 for word in sublist]
        
        uni_freq_dist = FreqDist(words)
        
        return uni_freq_dist
    
    def build_bi_con_prob_dist(self, sentence_list):
        """
        Returns a conditional probability distibution for 
        the bigrams in a list of sentences
        """
        
        bgrams = [bigram for sublist in 
                  [bigrams(['<s>'] + word_tokenize(sent.lower())) 
                  for sent in sentence_list] 
                  for bigram in sublist]
        
        bi_cfreq_dist = ConditionalFreqDist(bgrams)
        bi_cprob_dist = ConditionalProbDist(bi_cfreq_dist, 
                                            MLEProbDist)
        
        return bi_cprob_dist
    
    def build_tri_con_prob_dist(self, sentence_list):
        """
        Returns a conditional probability distibution for 
        the trigrams in a list of sentences
        """
        
        trgrams = [trigram for sublist in 
                   [trigrams(['<s>', '<s>'] + word_tokenize(sent.lower())) 
                   for sent in sentence_list] 
                   for trigram in sublist]
        
        ###################################
        ## Add Lidstone/Laplace smoothing##
        ###################################
        
        condition_pairs = (((w0, w1), w2) for w0, w1, w2 in trgrams)
        tri_cfreq_dist = FreqDist(trgrams)
        tri_cprob_dist = LaplaceProbDist(tri_cfreq_dist, 
                                             MLEProbDist)
        
        self.tri_cprob_dist = tri_cprob_dist
        
        return self.tri_cprob_dist
    
    
    def linguistic_features(self, sentence):
        """
        Returns an arrady containing num of right-branching nodes,
        #num of left-branching nodes, branching index, num of premodifiers,
        #num of postmodifiers, and modification index
        """
        
        sent = self.nlp(sentence)
        
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
        for token in sent:
            if token.i < token.head.i:
                left += 1
            elif token.i > token.head.i:
                right += 1
            if token.dep in self.modifiers and token.head.i > token.i:
                pre += 1
                mw_pre += len([child for child in token.subtree])
            elif token.dep in self.modifiers and token.head.i < token.i:
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
        for chunk in sent.noun_chunks:
            for token in chunk:
                if token.i < token.head.i:
                    np_left += 1
                elif token.i > token.head.i:
                    np_right += 1
                if token.dep in self.modifiers and token.head.i > token.i:
                    np_pre += 1
                    np_mw_pre += len([child for child in token.subtree])
                elif token.dep in self.modifiers and token.head.i < token.i:
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
        
    def function_word_densities(self, sentence):
        
        """
        Returns the sentence densities of function words, determiners,
        pronouns, prepositions, punctuations, auxiliaries, conjunctions,
        and wh-pronouns
        """
        
        sent = self.nlp(sentence)
        
        function = 0
        determiners = 0
        pronouns = 0
        prepositions = 0
        punctuations = 0
        auxiliaries = 0
        conjunctions = 0
        wh_pronouns = 0
        
        for token in sent:
            if (token.pos in self.function_pos 
                or token.tag_ in self.function_tag):
                function += 1
            if token.pos in [DET] or token.tag_ in ["DT"]:
                determiners += 1
            if (token.pos in [PRON] 
                or token.tag_ in ['PRP', 'PRP$', 'WP', 'WP$']):
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
        
        return [item / len(sent) for item in [function, 
                                            determiners, 
                                            pronouns, 
                                            prepositions, 
                                            punctuations, 
                                            auxiliaries, 
                                            conjunctions, 
                                            wh_pronouns]]
    
    def constituent_sizes(self, sentence):
        """
        Returns the number of words in noun phrases, adjectival phrases,
        prepositional phrases, and adverbial phrases
        """
    
        sent = self.nlp(sentence)
        
        np_lengths = []
        ajp_lengths = []
        pp_lengths = []
        avp_lengths = []
        
        for chunk in sent.noun_chunks:
            np_lengths.append(len(chunk))
            
        for token in sent:
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
                len(sent)]
        
    def perplexity(self, sentence):
        """
        Returns the per word sentence perplexity for unigram, bigram, 
        and trigram models for the base sentence and pos
        """
        
        sent = word_tokenize(sentence.lower())
        
        trgrams = [(["<s>", "<s>"] + sent)[n:n+3] for n in range(len(sent))]
        tri_sent_per = 2 ** sum([self.tri_cprob_dist[(trigram[0], trigram[1])].logprob(trigram[2]) for trigram in trgrams])
        
        return tri_sent_per
        
        #return [uni_sent_per,
        #        bi_sent_per,
        #        tri_sent_per,
        #        uni_pos_per,
        #        bi_pos_per,
        #        tri_sent_per]
        
        
        

        
        
@timeit
def build_and_evaluate(X, y, 
                       classifier=SGDClassifier,
                       outpath=None,
                       verbose=True):
    
    @timeit
    def build(classifier, X, y=None):
        if isinstance(classifier, type):
            #initializes the SGDClassifier?
            classifier = classifier()
            
        model = Pipeline([('preprocessor', NLTKPreprocessor()),
                          ('vectorizer', TfidfVectorizer(
                                  tokenizer=identity, 
                                  preprocessor=None,
                                  lowercase=False
                                  )),
                                  ('classifier', classifier)
                        ])
        
        model.fit(X, y)
        return model
    
    labels = LabelEncoder()
    y = labels.fit_transform(y)
        
    if verbose: print("Biulding for evaluation")
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
    model, secs = build(classifier, X_train, y_train)
    
    if verbose:
        print("Evaluation model fit in {:0.3f} seconds".format(secs))
        print("Classification Report:\n")
        
    y_pred = model.predict(X_test)
    print(clsr(y_test, y_pred, target_names=labels.classes_))
    
    if verbose:
        print("Building complete model and saving ...")
    model, secs = build(classifier, X, y)
    model.labels_ = labels
        
    if verbose:
        print("Complete model fit in {:0.3f} seconds".format(secs))
        
    if outpath:
        with open(outpath, 'wb') as f:
            pickle.dump(model, f)

        print("Model written out to {}".format(outpath))

    return model
        
        
        
        
        
        
        
        
        
        
        
        