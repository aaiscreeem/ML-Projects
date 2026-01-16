no_smoothing = {
    "method_name": "NO_SMOOTH",
    "n": 3,  # Default n-gram size
    "lowercase": True,
    "remove_punctuation": False,
    "unk_count": 1  # Default count for unknown words
} # 279 for n=3, 165 for n=2, 757 for n=4

add_k = {
    "method_name": "ADD_K",
    "n": 3,  # Default n-gram size
    "lowercase": False,
    "remove_punctuation": False,
    "unk_count": 1,  # Default count for unknown words
    "k": 8*1e-4  # Smoothing factor
}
# k= 1e-4, 3040, 1e-3, 2475, 1e-2, 3000



stupid_backoff = {
    "method_name" : "STUPID_BACKOFF",
    'alpha': 0.7,
    "n": 4,  # Default n-gram size
    "lowercase": False,
    "remove_punctuation": False,
    "unk_count": 1,  # Default count for unknown words
} # 63 for n=3, 152 for n=2, 38 for n=4

good_turing = {
    "method_name" : "GOOD_TURING",
    "n": 3,  # Default n-gram size
    "lowercase": False,
    "remove_punctuation": False,
    "unk_count": 1,  # Default count for unknown words
} # 9 for n=4, 118 for n=3, 5769 for n=2

interpolation = {
    "method_name" : "AddK",
    'lambdas': [0.4, 0.3, 0.3],
    "n": 5,  # Default n-gram size
    "lowercase": False,
    "remove_punctuation": False,
    "unk_count": 1,  # Default count for unknown words
    
}# 614 for n=3, 529 for n=2, 503 for n=4, 651 for n=5 

kneser_ney = {
    "method_name" : "KNESER_NEY",
    'discount': 0.2,
    "n": 4,  # Default n-gram size
    "lowercase": False,
    "remove_punctuation": False,
    "unk_count": 1,  # Default count for unknown words
}  

error_correction = {
    "internal_ngram_best_config" : {
        "method_name" : "GOOD_TURING",
        "n": 2,                      # Order of the n-gram model 
        "k": 8*1e-4,                      # Smoothing parameter for Add-K (if needed)
        "discount": 0.2,            # Discount factor for Kneser-Ney smoothing
        "backoff_weight": 0.7        # Weight for Stupid Backoff
    },
    "similarity_measure": "edit distance",    # Set similarity measure
    "log_coefficient": 2,            # Coefficient for log(P(S))
    "prob_threshold": 1,             # Threshold for accepting corrections
    # "error_rate": 5            # Number of words per error
}

