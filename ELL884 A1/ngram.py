import numpy as np
import pandas as pd
from typing import List
# import spacy
import contractions
import re
import math
import random



class NGramBase:
    def __init__(self):
        """
        Initialize basic n-gram configuration.
        :param n: The order of the n-gram (e.g., 2 for bigram, 3 for trigram).
        :param lowercase: Whether to convert text to lowercase.
        :param remove_punctuation: Whether to remove punctuation from text.
        """
        self.current_config = {}

        # change code beyond this point
        self.n = 3
        self.lowercase = False
        self.remove_punctuation = False
        self.count_dict={} # gram : count
        self.vocab={}  # word : count
        self.count_n_grams = 0 # total ngrams
        self.N = {} # N[a] is number of ngrams occuring a times
        self.total_words = 1
        self.unk_count = 1
        self.word_set = set()
        

    def method_name(self) -> str:

        return f"Method Name: {self.current_config['method_name']}"

    def fit(self, data: List[List[str]]) -> None:
        """
        Fit the n-gram model to the data.
        :param data: The input data. Each sentence is a list of tokens.
        """
        # build up vocab
        for sentence in data:
            for word in sentence:
                        if word in self.vocab:
                            self.vocab[word] += 1
                        else:
                            self.vocab[word] = 1
                        self.total_words += 1
        
        for word in self.vocab:
            self.word_set.add(word)
        # replace once occuring words with UNK and count them
        self.vocab['<UNK>'] = 1
        
        for sentence in data:
            for i in range(len(sentence)):
                if self.vocab[sentence[i]] == 1:
                    sentence[i] = '<UNK>'
                    self.vocab['<UNK>'] += 1
        
        # print("UNK done")   
        for sentence in data:
            for gram_size in range(1, self.n + 1):  # Count all grams
                for i in range(len(sentence) - gram_size + 1):
                    ngram = tuple(sentence[i : i + gram_size])  # Tuples are immutable and hashable
                    if ngram in self.count_dict:
                        self.count_dict[ngram] += 1
                    else:
                        self.count_dict[ngram] = 1
                    if gram_size == self.n :
                        self.count_n_grams += 1
                        
        for ngram in self.count_dict:
            if len(ngram) == self.n:
                count = self.count_dict[ngram]
                if count in self.N:
                    self.N[count] += 1
                else:
                    self.N[count] = 1
        
        self.unk_count = self.vocab['<UNK>']
              
        return
    
    def probability(self, word, context):
        if self.n > 1 and context in self.count_dict:
        # If (context, word) exists, return its probability
            if (context + (word,)) in self.count_dict:
                return self.count_dict[(context + (word,))] / self.count_dict[context]

            # Otherwise, fall back to UNK if present
            elif (context + ("<UNK>",)) in self.count_dict:
                return self.count_dict[(context + ("<UNK>",))] / self.count_dict[context]
            
        return self.vocab.get(word, self.unk_count)/self.total_words # unigram prob


    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text.
        :param text: The input text.
        :return: The list of tokens.
        """
        tokens = []
        sentences = self.split_into_sentences(text)
        for sentence in sentences:
            words = ['<s>'] * (self.n - 1) + sentence.split()
            tokens.extend(words + ['</s>'])

        
        return tokens
        raise NotImplementedError

    def prepare_data_for_fitting(self, data: List[str], use_fixed = True) -> List[List[str]]:
        """
        Prepare data for fitting.
        :param data: The input data.
        :return: The prepared data.
        """
        processed = []
        if not use_fixed:
            for text in data:
                processed.append(self.tokenize(self.preprocess(text)))
        else:
            for text in data:
                processed.append(self.fixed_tokenize(self.fixed_preprocess(text)))

        return processed

    def update_config(self, config) -> None:
        
        self.current_config = config
        if "n" in config:
            self.n = config["n"]
        if "lowercase" in config:
            self.lowercase = config["lowercase"]
        if "remove_punctuation" in config:
            self.remove_punctuation = config["remove_punctuation"]
        if "unk_count" in config:
            self.unk_count = config["unk_count"]
        
        


    def selective_expand_contractions(self,text: str) -> str:
        def is_grammatical_contraction(contraction, expanded):
            """
            Returns True if the contraction is a grammatical one (verb contraction or negation),
            otherwise False.
            """
            words = expanded.split()
            if len(words) > 1:
                # Expand contractions only if they contain verbs or negations
                if any(w in words for w in ["am", "is", "are", "was", "were", "have", "has", "had",
                                            "will", "would", "shall", "should", "do", "does", "did",
                                            "can", "could", "must", "might", "may", "not", "n't"]):
                    return True
            return False

        # Use regex to identify contractions (words with an apostrophe)
        contraction_pattern = re.compile(r"\b\w+'\w+\b")
        
        def replace_match(match):
            contraction = match.group(0)
            expanded = contractions.fix(contraction)
            return expanded if is_grammatical_contraction(contraction, expanded) else contraction

        return contraction_pattern.sub(replace_match, text)

        



    def preprocess(self, text: str) -> str:
        """
        Preprocess text before n-gram extraction.
        :param text: The input text.
        :return: The preprocessed text.
        """
#         remove the characters besides alphabets, digits, !, $, %, &, brackets, ", ?, comma, period, ', newline, 
#         also create a space before and after !, ?, brackets, ", comma ONLY IF space does not already exist
        
        def clean_text(text):
            allowed_chars = r"[^a-zA-Z0-9!$%&\(\)\[\]\{\}\"',\?\.\s'\/\-]"
            text = re.sub(allowed_chars, '', text)

            spaced_chars = r"[!\"?\[\]\(\),-]"  # Characters to add spaces around

            # Add space before if there's no space before it                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
            text = re.sub(r"(?<!\s)([" + spaced_chars[1:-1] + r"])", r" \1", text)

            # Add space after if there's no space after it
            text = re.sub(r"([" + spaced_chars[1:-1] + r"])(?!\s)", r"\1 ", text)
            text = re.sub(r'(?<=\s)(\.)((?!\s))', r'. ', text) # add space after period only if there is a space just before it
            return re.sub(r'\s+', ' ', text).strip()
        
        text = self.selective_expand_contractions(text)
        
        if self.lowercase:
            if self.remove_punctuation:
                text = re.sub(r'[^\w\s]', '', text)
            text = clean_text(text)
            text = text.lower()
        else:
            if self.remove_punctuation:
             text = re.sub(r'[^\w\s]', '', text)
            text = clean_text(text)
        # print("preprocessed")
        return text
            
        
        raise NotImplementedError
    
    def generate_text(self,max_length=15)->str:
            answer = ["<s>"]*(self.n-1)
            for i in range(max_length):
                vals = []
                words = []
                prev = 0.0
                sum = 0.0
                for word in self.vocab:
                    words.append(word)
                    ngram = answer[-(self.n-1):]+[word]
                    prob = self.probability(word, tuple(ngram[:-1]))
                    sum += prob
                    vals.append(prob)
                vals = [v/sum for v in vals]
                for i in range(len(vals)):
                    prev += vals[i]
                    vals[i] = prev
                num = random.random()
                l , r = 0 , len(vals)-1
                final_word = "<s>"
                while l <= r:
                    mid = (l+r)//2
                    if vals[mid] >= num:
                        final_word = words[mid]
                        r = mid-1
                    else:
                        l = mid+1
                answer.append(final_word)
            answer = answer[self.n-1:]
            print("answer ;",answer)
            return ' '.join(answer)

    def fixed_preprocess(self, text: str) -> str:
        """
        Removes punctuation and converts text to lowercase.
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def fixed_tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the input text by splitting at spaces.
        """
        return text.split()



    def split_into_sentences(self, text: str) -> List[str]:
        """
        Splits a given text into sentences, handling basic cases and edge cases like abbreviations, decimals, and ellipses.
        
        Args:
            text (str): The input text to split into sentences.
        
        Returns:
            List[str]: A list of sentences.
        """
        # Define common abbreviations that should not cause a sentence split
        abbreviations = {
            "Sen.", "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Inc.", "e.g.", "i.e.",
            "vs.", "Jr.", "Sr.", "Atty.", "Gen.", "Gov.", "U.S.A.", "U.S.", "Ph.D.", "M.D."
        }

        # Preprocess the text: replace newlines with spaces and normalize spaces
        text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
        text = re.sub(r'\s+', ' ', text)  # Normalize multiple spaces to a single space

        # Use regex to split sentences based on punctuation marks
        sentence_delimiters = re.compile(r'(?<=[.!?])\s+')  # Lookbehind for .!? followed by spaces
        sentences = sentence_delimiters.split(text)

        # Handle edge cases: abbreviations, decimal numbers, and ellipses
        final_sentences = []
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            # Check if the sentence ends with an abbreviation or decimal
            if (any(sentence.endswith(abbr) for abbr in abbreviations) or
                re.search(r'\d+\.\d+$', sentence)):  # Decimal numbers
                # Merge with the next sentence
                if i + 1 < len(sentences):
                    sentence += " " + sentences[i + 1]
                    i += 1  # Skip the next sentence
            # Handle ellipses (...)
            if sentence.endswith('...'):
                # Merge with the next sentence
                if i + 1 < len(sentences):
                    sentence += " " + sentences[i + 1]
                    i += 1  # Skip the next sentence
            final_sentences.append(sentence)
            i += 1

        return final_sentences

    def perplexity(self, text: str) -> float:
        """
        Compute the perplexity of the model given the text.
        :param text: The input text.
        :return: The perplexity of the model.
        """
        processed_sentences = self.prepare_data_for_fitting(self.split_into_sentences(text))
        # tokens = self.tokenize(self.preprocess(text))
        log_prob = 0
        total = 0
        for tokens in processed_sentences:
            for i in range(len(tokens)):
                if tokens[i] == '</s>' or tokens[i] == '<s>':
                    continue
                if tokens[i] not in self.vocab or self.vocab[tokens[i]] == 1:
                    tokens[i] = '<UNK>'
                prob = 1
                if self.n == 1:
                    prob = self.vocab.get(tokens[i], self.unk_count)/self.total_words
                else: 
                    context = tuple(tokens[i-self.n+1:i])
                    prob = self.probability(tokens[i], context)
                if prob<=0:
                    return float('inf')
                log_prob += math.log(prob)
                total += 1
        return math.exp((-log_prob/total)) 

# if __name__ == "__main__":
#     tester_ngram = NGramBase()
#     test_sentence = "The Fulton County Grand Jury said Friday an investigation of Atlanta's recent primary election produced `` no evidence '' that any irregularities took place .The jury further said in term-end presentments that the City Executive Committee , which had over-all charge of the election , `` deserves the praise and thanks of the City of Atlanta '' for the manner in which the election was conducted .The September-October term jury had been charged by Fulton Superior Court Judge Durwood Pye to investigate reports of possible `` irregularities '' in the hard-fought primary which was won by Mayor-nominate Ivan Allen Jr. . `` Only a relative handful of such reports was received '' , the jury said , `` considering the widespread interest in the election , the number of voters and the size of this city '' . The jury said it did find that many of Georgia's registration and election laws `` are outmoded or inadequate and often ambiguous '' . It recommended that Fulton legislators act `` to have these laws studied and revised to the end of modernizing and improving them '' ."
#     processed_sentences = tester_ngram.prepare_data_for_fitting(tester_ngram.split_into_sentences(test_sentence))
#     # for sentence in processed_sentences:
#     #     print(sentence)
#     tester_ngram.fit(processed_sentences)
#     # print(tester_ngram.N)
#     # print(tester_ngram.count_n_grams)
#     # print(tester_ngram.vocab)
#     print(tester_ngram.perplexity("The jury did"))
    
    
    
    
    
    
    
    
