from smoothing_classes import *
from config import error_correction
import numpy as np
import pandas as pd
import math
from typing import List

# in the given data errors are not due to phonetic similarity or OCR, most seem to be typing error
# we want word S which maximizes P(S|O) = P(O|S)P(S)/P(O) 
# now P(S) from ngram data and P(O) is constant. to find P(O|S): 
# We'll consider Damerau-Levenshtein edit distance, which also includes transposition besides the other 3
# we can have 26*26 confusion matrix for probability of writing i in place of j from google or somewhere. 
# We can assume probability of missing a letter, swapping 2 letters and writing an extra letter to be a, b & c (hyperparameters) . 
# can completely ignore punctuations here
# only consider words till edit distance of a threshold
# can assume at max len(sentence)/e errors where e is hyperparameter
# use a blacklist, forbidding certain tokens (like numbers, punctuation, and single letter words) from being changed.
# can choose a threshold probability difference only beyond which correction is done
# can choose log(P(O|S))+ lambda * log(P(S)) hyperparameter


# Additionally:
# word boundary errors
# final rule based processing
# OOV correct words


def jaro_similarity(s1, s2):
    """Compute the Jaro Similarity between two words."""
    s1, s2 = s1.lower(), s2.lower()
    len_s1, len_s2 = len(s1), len(s2)

    if len_s1 == 0 and len_s2 == 0:
        return 1.0

    match_distance = max(len_s1, len_s2) // 2 - 1
    s1_matches = [False] * len_s1
    s2_matches = [False] * len_s2

    matches = 0
    transpositions = 0

    # Find matches
    for i in range(len_s1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len_s2)

        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] == s2[j]:
                s1_matches[i] = s2_matches[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    # Count transpositions
    k = 0
    for i in range(len_s1):
        if s1_matches[i]:
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

    transpositions /= 2

    jaro_sim = (matches / len_s1 + matches / len_s2 + (matches - transpositions) / matches) / 3
    return jaro_sim


def jaro_winkler_similarity(s1, s2, p=0.1):
    """Compute the Jaro-Winkler similarity."""
    jaro_sim = jaro_similarity(s1, s2)
    prefix_length = 0

    for i in range(min(len(s1), len(s2))):
        if s1[i] == s2[i]:
            prefix_length += 1
        else:
            break
        if prefix_length == 4:  # Jaro-Winkler considers at most first 4 chars
            break

    return jaro_sim + (prefix_length * p * (1 - jaro_sim))


def sequence_matcher_ratio(s1, s2):
    """Compute the similarity using Ratcliff/Obershelp algorithm."""
    s1, s2 = s1.lower(), s2.lower()

    def lcs_length(s1, s2):
        """Find longest contiguous matching subsequence."""
        max_len = 0
        for i in range(len(s1)):
            for j in range(len(s2)):
                l = 0
                while (i + l < len(s1)) and (j + l < len(s2)) and (s1[i + l] == s2[j + l]):
                    l += 1
                max_len = max(max_len, l)
        return max_len

    lcs = lcs_length(s1, s2)
    if lcs == 0:
        return 0.0

    return (2 * lcs) / (len(s1) + len(s2))


def lev_edit_prob(word1: str, word2: str, p_sub: float, p_ins: float, p_del: float, p_trans: float) -> float:
    """
    Computes the sum of log probabilities of transforming word1 into word2
    using edit operations with given probabilities.

    Substitutions have probability p_sub,
    Insertions have probability p_ins,
    Deletions have probability p_del,
    Transpositions have probability p_trans.
    """
    import numpy as np
    n, m = len(word1), len(word2)
    dp = np.full((n + 1, m + 1), float('inf'))

    dp[0][0] = 1

    for i in range(n + 1):
        for j in range(m + 1):
            if i > 0:
                dp[i][j] = min(dp[i][j], dp[i - 1][j] * p_del)  # Deletion
            if j > 0:
                dp[i][j] = min(dp[i][j], dp[i][j - 1] * p_ins)  # Insertion
            if i > 0 and j > 0:
                cost = 1 if word1[i - 1] == word2[j - 1] else p_sub  # Substitution
                dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] * cost)
            if i > 1 and j > 1 and word1[i - 1] == word2[j - 2] and word1[i - 2] == word2[j - 1]:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] * p_trans)  # Transposition

    return dp[n][m]








class SpellingCorrector:

    def __init__(self):

        self.correction_config = error_correction
        self.internal_ngram_name = self.correction_config['internal_ngram_best_config']['method_name']

        if self.internal_ngram_name == "NO_SMOOTH":
            self.internal_ngram = NoSmoothing()
        elif self.internal_ngram_name == "ADD_K":
            self.internal_ngram = AddK()
        elif self.internal_ngram_name == "STUPID_BACKOFF":
            self.internal_ngram = StupidBackoff()
        elif self.internal_ngram_name == "GOOD_TURING":
            self.internal_ngram = GoodTuring()
        elif self.internal_ngram_name == "INTERPOLATION":
            self.internal_ngram = Interpolation()
        elif self.internal_ngram_name == "KNESER_NEY":
            self.internal_ngram = KneserNey()

        self.internal_ngram.update_config(self.correction_config['internal_ngram_best_config'])
        
        self.blacklist =['<s>', '</s>',',','!','.','?','(',')','[',']','{','}','<UNK>']
        
        self.l = self.correction_config.get('log_coefficient', 1)
        self.threshold = self.correction_config.get('prob_threshold', 0)
        # self.e = self.correction_config.get('error_rate', 4)
        self.similarity_measure = self.correction_config.get('similarity_measure', 'jaro')


    def fit(self, data: List[str]) -> None:
        """
        Fit the spelling corrector model to the data.
        :param data: The input data.
        """
        processed_data = self.internal_ngram.prepare_data_for_fitting(data, use_fixed=True)
        self.internal_ngram.fit(processed_data)
        
    def edit_prob(self, word1, word2):
        if self.similarity_measure == 'jaro':
            return jaro_winkler_similarity(word1, word2)
        elif self.similarity_measure == 'edit distance':
            return lev_edit_prob( word1, word2, 0.027,0.004, 0.016, 0.003)
        return sequence_matcher_ratio( word1, word2)
    
    
    def generate_edits(self, word, n):
        """
        Generate all possible words within edit distance n that exist in self.internal_ngram.word_set.
        """
        if n == 0:
            return {word} if word in self.internal_ngram.word_set else set()
        
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        
        # All possible edits (deletion, insertion, substitution, transposition)
        deletes = [L + R[1:] for L, R in splits if R]
        inserts = [L + c + R for L, R in splits for c in letters]
        substitutes = [L + c + R[1:] for L, R in splits if R for c in letters]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]

        # Combine all edits and filter for words in self.internal_ngram.word_set
        edits = set(deletes + inserts + substitutes + transposes)
        valid_edits = {e for e in edits if e in self.internal_ngram.word_set}

        # Recursively generate further edits if needed
        if n > 1:
            return valid_edits | {e2 for e1 in valid_edits for e2 in self.generate_edits(e1, n - 1)}
        
        return valid_edits
        
    def correct(self, text: List[str]) -> List[str]:
        """
        Correct the input text.
        :param text: The input text.
        :return: The corrected text.
        """
        corrected_text = []
        count_change = 0
        avg_log_prob = 0
        avg_edit_prob = 0
        avg_ngram_prob = 0
        for sentence in text:
            
            words = self.internal_ngram.tokenize(sentence)
            replacements = []
            for i in range(len(words)):
                if words[i] in self.blacklist or words[i][0].isdigit() :
                    continue
                curr_word = words[i]
                context = tuple(words[i-self.internal_ngram.n+1:i])
                original_val = self.l*math.log(self.internal_ngram.probability(words[i], context))
                if curr_word not in self.internal_ngram.word_set:
                    original_val = self.l*math.log(1e-8) # can be changed
                # else:
                #     continue
                curr_val = original_val
                edits = self.generate_edits(words[i], 2)
                for s in edits:
                    if len(s) == 1:
                        continue
                    ngram_prob = self.internal_ngram.probability(s, context)
                    if ngram_prob == 0:
                         ngram_prob = 1e-7
                    log_prob = math.log(max(self.edit_prob(words[i], s),1e-6)) + self.l * math.log(ngram_prob)
                    if  log_prob > curr_val:
                        curr_val = log_prob
                        curr_word = s
                replacements.append( [curr_val - original_val, curr_word, i] )
            replacements = sorted(replacements, key=lambda x: x[0], reverse=True)
            for i in range(min(len(replacements),2)):
                if replacements[i][0] > self.threshold:
                    avg_log_prob += replacements[i][0]
                    count_change += 1
                    words[replacements[i][2]] = replacements[i][1]
            corrected_text.append(' '.join([word for word in words if word not in {'<s>', '</s>', '<UNK>'}]))
        # print("count_change")
        # print(count_change)  
        # print("avg_log_prob")
        # print(avg_log_prob/(count_change+1))
        # print("avg_ngram_prob")
        # print(avg_ngram_prob/3181) -7.2
        # print("test_words") 3181 
        # print("words_not_in_vocab") 252
        return corrected_text
