import sys
from collections import defaultdict
import math
import random
import os
import os.path


def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  

def get_ngrams(sequence, n):
    if n < 1: 
        raise ValueError(("n must be at least 1"))

    # start_token = ['START'] * (n-1)
    if n != 1:
        start_token = ["START"] * (n-1)
        stop_token = ["STOP"]
        padded_sequence = start_token + sequence + stop_token
    else:
        padded_sequence = sequence

    ngrams = []
    for i in range(len(padded_sequence)-n+1):
        ngrams.append(tuple(padded_sequence[i : i+n]))

    return ngrams                            


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        self.num_sentences = 0
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)
        self.total_word_count = sum([value for value in self.unigramcounts.values()])
        


    def count_ngrams(self, corpus):
        self.unigramcounts = defaultdict(int) # defaultdict or Counter 
        self.bigramcounts = defaultdict(int) 
        self.trigramcounts = defaultdict(int)

        
        for sentence in corpus:
            for unigram in get_ngrams(sentence, 1):
                self.unigramcounts[unigram] += 1
            for bigram in get_ngrams(sentence, 2):  
                self.bigramcounts[bigram] += 1
            for trigram in get_ngrams(sentence, 3):
                self.trigramcounts[trigram] += 1
            self.num_sentences += 1


    def raw_trigram_probability(self,trigram):
        if trigram[0] == "START" and trigram[1] == "START":
            num_of_sentences_starting_with_word = self.bigramcounts.get(("START",trigram[2]),0)/self.num_sentences
            return num_of_sentences_starting_with_word
        trigram_count = self.trigramcounts.get(trigram,0)
        if trigram_count == 0:
            return 1/len(self.lexicon)
        bigram_count = self.bigramcounts.get(trigram[:-1],0)
        if bigram_count == 0: 
            return self.raw_unigram_probability(trigram[2])
        else:
            return trigram_count/bigram_count
        

    def raw_bigram_probability(self, bigram):
        bigram_count = self.bigramcounts.get(bigram,0)
        unigram_count = self.unigramcounts.get(bigram[0], 0)
        if unigram_count == 0:
            return self.raw_unigram_probability(bigram[1])
        else:
            return bigram_count/unigram_count
        

    
    def raw_unigram_probability(self, unigram):
        unigram_count = self.unigramcounts.get(unigram,0)
        if unigram_count == 0:
            return 1/len(self.lexicon)
        return unigram_count/self.total_word_count


    # def generate_sentence(self,t=20): 
        # sentence = ["START", "START"]
        # while len(sentence) < t+2 whatever :(   tired will come back
                 

    def smoothed_trigram_probability(self, trigram):
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        raw_tri_prob = self.raw_trigram_probability(trigram)
        raw_bi_prob  = self.raw_bigram_probability((trigram[1], trigram[2]))
        raw_uni_prob = self.raw_unigram_probability((trigram[2],)) 
        return lambda1 * raw_tri_prob + lambda2 * raw_bi_prob + lambda3 * raw_uni_prob
            
        
    def sentence_logprob(self, sentence):
        trigrams_sentence = get_ngrams(sentence,3)
        logprob = 0.0
        for trigram in trigrams_sentence:
            logprob += math.log2(self.smoothed_trigram_probability(trigram))
        return logprob

    def perplexity(self, corpus):
        logprob = 0.0
        num = 0
        for sentence in corpus:
            logprob += self.sentence_logprob(sentence)
            num += len(sentence) + 1

        perplexity = 2 ** (-logprob/num)
        return perplexity