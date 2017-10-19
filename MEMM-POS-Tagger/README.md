# MEMM-POS-Tagger

## Overview

In this project, I will develop a Part-of-Speech (POS) Tagger using first-order Maximum Entropy Markov Models (MEMM) on custom-engineered fetures.

## Performance

The POS tagger developed in this project can predict POS tags of words in new sentences with **95.7%** test accuracy (on a test set of 15K words), after training on the Penn Treebank [Marcus et al., 1993] which consists of around 1.2 million words from the the Wall Street Journal (WSJ).

## Model and Algorithm

This model consists of two parts: (1) an n-gram model (aggregated with additional custom features) for part of speech sequences; and (2) a likelihood distribution model of part of speech tags for words. 

These parts are combined using Bayes Theorem, and the Viterbi algorithm is used to find the most probable part of speech sequence given a set of words. The tagger developed in this project also provides a greedy algorithm as an alternative to the Viterbi algorithm.

## Hyper-Parameters

The hyper-parameters include:

- `l2-regularization strength`: the l2-regularization strenght used for the logistic regression by [sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)
- `minimum feature count`: the minimum number of occurences in the training data for the feature to be used 

During the model training, a `l2-regularization strength` of `0.9`, and a `minimum feature count` of `2` are chosen and used.

These hyper-parameters are tunned as the best performing, after a grid search of the following parameter combinations:

- l2 regularization strength: 1, 0.95, 0.9, 0.85, 0.8
- minimum features count: 1, 2, 3, 5


## Feature Engineering

The following sections describe the custom engineered features of the tagger, and an abalation test evaluating the importance of each feature.


### Explanation of Features

There are a number of custom engineered features used in the final model, and here is a brief description of them:

1. **`UNIGRAM_W`**: The word `w_t` itself
2. **`PREVIOUS_TAG_POS`**: The `POS` tag assigned to the previous word `w_{t-1}`
3. **`PREFIX_W`**: The prefix of word `w_t`, defined as the first 3 characters of the word.
4. **`SUFFIX__W`**: The suffix of word `w_t`, defined as the last 2 characters of the word. (_Having this feature is helpful for cases like plural nouns, third-person/past-tense verbs, comparative adj/adv, etc._) 
5. **`BIGRAM_W1_W2`**:
	- _Condition on Past_: The first type of bigram is the word `w_1=w_t` conditioned on its previous word `w_2=w_{t-1}`
	- _Condition on Future_: The second type of bigram is the word `w_1=w_t` conditioned on its subsequent word `w_2=w_{t+1}`
6. **`TRIGRAM_W1_W2_W3`**: 
	- The trigram is the word `w_2=w_t` conditioned on its previous word `w_1=w_{t-1}` and its subsequent word `w_3=w_t`. 
	- This choice is determined by examing the performance of the following three possible bigram windows: 
		-  `w_{t-2}, w_{t-1}, w_t`
		- `w_{t-1}, w_t, w_{t+1}`
		- `w_t, w_{t+1}, w_{t+2}` 
	- It turns out using ONLY the middle version produces the best performance as it contains the information both about the past and the future.
7. **`FIRST_WORD_IN_SEQUENCE`**: true if the word `w_t` is the first word in the sequence
8. **`LAST_WORD_IN_SEQUENCE `**: true if the word `w_t` is the last word in the sequence
9. **`NUMERIC`**: true if the word `w_t` is a number, with the possible inclusion of punctuation symbols and character `s`. (_Having this feature helps detect cardinal numbers, for cases like 1970s, $10,224_)
10. **`FIRST_UPPER `**: true if the first character of the word `w_t` is capitalized. (_Having this feature helps detect name entities._)


### Ablation Test Results

Using the same hyper-parameters, 10% of the training data, and 4.5K test data, I conducted a series of **ablation tests** as follows to determine the importance of each feature. 

| Feature | Accuracy | Abalation |
| ------|:------:| :------:|
| Full Model | **0.949** | 0.949 |
| Previous Tag | 0.928 | -0.021 |
| Prefix | 0.941 | -0.008 |
| Suffix | 0.912 | -0.037 |
| Numeric | 0.939 | -0.010 |
| First Letter Capitalization | 0.919 | -0.030 |
| First Word In Sequence | 0.936 | -0.013 |
| Last Word In Sequence | 0.948 | -0.001 |
| Word Unigram | 0.865 | -0.084 |
| Word Bigram (with past word) | 0.941 | -0.008 |
| Word Bigram (with future word) | 0.934 | -0.015 |
| Word Trigram | 0.940 | -0.009 |



