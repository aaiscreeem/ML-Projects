from ngram import NGramBase
from config import *
import numpy as np
import pandas as pd
from typing import List
from collections import defaultdict

# <UNK> tokens should be implemented

class NoSmoothing(NGramBase):

    def __init__(self):

        super(NoSmoothing, self).__init__()
        self.update_config(no_smoothing)

class AddK(NGramBase):

    def __init__(self):
        super(AddK, self).__init__()
        self.update_config(add_k)
        self.k = self.current_config["k"]
    
    def probability(self, word, context):
        # context is a tuple
        if self.n > 1 and context in self.count_dict:
            return (self.count_dict.get(context + (word,), 0) + self.k) / (self.count_dict[context] + self.k*len(self.vocab))
        
        return self.vocab.get(word, self.unk_count)/self.total_words # unigram prob      
    

class StupidBackoff(NGramBase):

    def __init__(self):
        super(StupidBackoff, self).__init__()
        self.update_config(stupid_backoff)
        self.alpha = self.current_config["alpha"]
    
    def probability(self, word, context):
        # context is a tuple 
        ans = 1
        for i in range(self.n-1):
            reduced_context = context[i:]
            if (reduced_context + (word,)) in self.count_dict:
                return ans*self.count_dict.get(reduced_context + (word,)) / self.count_dict.get(reduced_context, 1)
            ans *= self.alpha
        return self.vocab.get(word, self.unk_count)/self.total_words # unigram probability
            
 

class GoodTuring(NGramBase):

    def __init__(self):
        super(GoodTuring, self).__init__()
        self.update_config(good_turing)  # Assume good_turing is defined elsewhere

    def probability(self, word, context):
        """
        Returns a nonzero probability for any n-gram.
        """
        # Use <UNK> for unknown words.
        if word not in self.vocab:
            word = "<UNK>"
        ngram = context + (word,)
        count = self.count_dict.get(ngram, 0)
        total_ngrams = self.count_n_grams
        
        # A small epsilon to ensure we never return zero probability.
        epsilon = 1e-7       
        # Protect against division by zero.
        if total_ngrams == 0:
            return epsilon
        if count == 0:
            N1 = self.N.get(1, 0)
            if N1 == 0:
                # Fallback: evenly distribute a tiny probability mass among all
                # possible n-grams using the vocabulary size.
                fallback_prob = 1 / (total_ngrams * max(len(self.vocab), 1))
                return fallback_prob if fallback_prob > 0 else epsilon
            else:
                prob = N1 / total_ngrams
                return prob if prob > 0 else epsilon


        Nc = self.N.get(count, 0)          # Count-of-count for the observed count.
        Nc_plus1 = self.N.get(count + 1, 0)  # Count-of-count for count+1.

        if Nc == 0 or Nc_plus1 == 0:
            # Fallback to maximum likelihood if we lack sufficient data for adjustment.
            prob = count / total_ngrams
            return prob if prob > 0 else epsilon

        # Compute the adjusted count: c* = (c+1) * (N_{c+1} / N_c)
        adjusted_count = (count + 1) * (Nc_plus1 / Nc)
        prob = adjusted_count / total_ngrams
        return prob if prob > 0 else epsilon

        
 

class Interpolation(NGramBase):

    def __init__(self):
        super(Interpolation, self).__init__()
        self.update_config(interpolation)
        self.l1 = self.current_config["lambdas"][0]
        self.l2 = self.current_config["lambdas"][1]
        self.l3 = self.current_config["lambdas"][2]
        
    def probability(self, word, context):
        # context is a tuple of size n-1
        ans = 0
        temp = [self.l1, self.l2, self.l3]
        if self.n == 1:
            return self.vocab.get(word, self.unk_count)/self.total_words
        if self.n == 2:
            if context in self.count_dict:
                ans += (self.l2/(self.l2+self.l3))* self.count_dict.get(context + (word,), 0) / self.count_dict[context]
            ans += (self.l3/(self.l2+self.l3))* self.vocab.get(word, self.unk_count)/self.total_words
            return ans
        
        if self.n == 3:
            if context in self.count_dict:
                ans += self.l1* self.count_dict.get(context + (word,), 0) / self.count_dict[context] 
            if context[1:] in self.count_dict:
                ans += self.l2* self.count_dict.get(context[1:] + (word,), 0) / self.count_dict[context[1:]] 
            ans += self.l3* self.vocab.get(word, self.unk_count)/self.total_words
            return ans        
            
                
        for i in range(3):
                reduced_context = context[i:]
                if reduced_context in self.count_dict:
                    ans += temp[i]*self.count_dict.get(reduced_context + (word,), 0) / self.count_dict[reduced_context]
        if ans == 0:
            return self.vocab.get(word, self.unk_count)/self.total_words
        return ans
                
class KneserNey(NGramBase):
    def __init__(self):
        super(KneserNey, self).__init__()
        self.update_config(kneser_ney)
        self.d = self.current_config["discount"]
        # Track continuation counts for all n-gram orders: {order: {word: set(contexts)}}
        self.continuation_counts = defaultdict(lambda: defaultdict(set))

    def fit(self, data: List[List[str]]) -> None:
        """
        Fit the n-gram model to the data while computing Kneser-Ney statistics.
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
        # print("hi2")
        
        for sentence in data:
            for i in range(len(sentence)):
                if self.vocab[sentence[i]] == 1:
                    sentence[i] = '<UNK>'
                    self.vocab['<UNK>'] += 1
        # print("hi3")
        # Second pass: Count n-grams and track continuation counts for Kneser-Ney
        self.continuation_counts.clear()
        for sentence in data:
            for gram_size in range(1, self.n + 1):
                for i in range(len(sentence) - gram_size + 1):
                    ngram = tuple(sentence[i:i + gram_size])
                    self.count_dict[ngram] = self.count_dict.get(ngram, 0) + 1

                    # Update continuation counts for Kneser-Ney
                    if gram_size >= 1:
                        context = ngram[:-1] if gram_size > 1 else ()
                        word = ngram[-1]
                        self.continuation_counts[gram_size][word].add(context)
        
        
        # Track counts of n-grams (for discounting in modified Kneser-Ney)
        self.N.clear()
        for ngram, count in self.count_dict.items():
            if len(ngram) == self.n:
                self.N[count] = self.N.get(count, 0) + 1

        self.unk_count = self.vocab['<UNK>']

    
    def probability(self, word: str, context) -> float:
        """
        Compute Kneser-Ney probability iteratively (no recursion) and handle zero probabilities.
        """
        # Replace unknown words in word and context
        word = word if word in self.word_set else '<UNK>'
        processed_context = tuple(
            c if c in self.word_set else '<UNK>' 
            for c in context
        )
        
        # Precompute UNK probability: P(UNK) = count(UNK) / vocab_size
        unk_prob = self.unk_count / len(self.vocab) if len(self.vocab) > 0 else 0.0
        
        # Initialize with unigram probability (order=1)
        current_prob = len(self.continuation_counts[2].get(word, set()))  # Continuation count
        total_continuations = sum(len(ctxs) for ctxs in self.continuation_counts[2].values())
        current_prob = current_prob / total_continuations if total_continuations > 0 else 0.0
        
        # print(1)
        
        # Fallback to UNK probability if unigram is zero
        current_prob = current_prob if current_prob > 0 else unk_prob
        
        # Iterate through higher orders (bigram, trigram, etc.)
        for order in range(2, len(processed_context) + 2):
            # Get context for this order (last 'order-1' tokens)
            current_order_context = processed_context[-(order-1):] if order > 1 else ()
            
            # Full n-gram = context + word
            full_ngram = current_order_context + (word,)
            
            # Get counts
            ngram_count = self.count_dict.get(full_ngram, 0)
            context_count = self.count_dict.get(current_order_context, 0)
            
            if context_count == 0:
                # Backoff: use lower-order probability without modification
                higher_order_prob = current_prob
            else:
                # Compute discounted probability
                discounted = max(ngram_count - self.d, 0.0) / context_count
                
                # Compute lambda: (d / c(context)) * num_unique_successors
                num_successors = sum(
                    1 for w in self.continuation_counts[order] 
                    if current_order_context in self.continuation_counts[order][w]
                )
                lambda_weight = (self.d / context_count) * num_successors
                
                # Interpolate
                higher_order_prob = discounted + lambda_weight * current_prob
            
            # Fallback to UNK probability if final result is zero
            current_prob = higher_order_prob if higher_order_prob > 0 else unk_prob
            
        
        return current_prob



      
from sklearn.model_selection import train_test_split

def train_and_validate_ngram_model(paths: List[str], ngram_model):
    training_text = []
    # Load training datasets
    for file in paths:
        with open(file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            training_text.append(line.strip())
    
    # Split the data into train and validation sets 
    train_data, val_data = train_test_split(training_text, test_size=0.2, random_state=42)
    
    # Prepare and train the n-gram model
    processed_train = ngram_model.prepare_data_for_fitting(train_data)
    # print("reached")
    ngram_model.fit(processed_train)
    # print("hi")
    
    # Calculate perplexity on validation data
    processed_val = ngram_model.prepare_data_for_fitting(val_data)
    val_perplexity = ngram_model.perplexity(' '.join(val_data))
    for _ in range(3):
        ngram_model.generate_text()
    
    print(f"Validation Perplexity: {val_perplexity}")
    return val_perplexity


# tester_ngram1 = StupidBackoff()
# tester_ngram2 = KneserNey()
# tester_ngram3 = StupidBackoff()
# paths = ['./data/train1.txt', './data/train2.txt']
# train_and_validate_ngram_model(paths, tester_ngram1)
# train_and_validate_ngram_model(paths, tester_ngram1)
# train_and_validate_ngram_model(paths, tester_ngram2)
# train_and_validate_ngram_model(paths, tester_ngram3)

