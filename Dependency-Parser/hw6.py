import sys,re
import numpy as np
from sklearn import linear_model
from scipy import sparse
from collections import Counter
from sklearn import grid_search
from sklearn import cross_validation

L2_REGULARIZATION_STRENGTH = 1e0
MINIMUM_FEAT_COUNT=5

# dictionary of feature: index
# feature = a dictionary of feature name: value
feature_vocab={}

# reverse_features[index] = feature
reverse_features=[]

# dictionary of label: index, where
# label = SHIFT, LEFTARC_DEPENDENCY_LABEL, RIGHTARC_DEPENDENCY_LABEL
label_vocab={}

# reverse_labels[index] = label
reverse_labels=[]

# number/ID of features
fmax = 0

# ============================================================
# TO BE IMPLEMENTED
# ============================================================
# Question 1
def is_projective(toks):
	"""
	params: toks is a list of (idd, tok, pos, head, lab) for a sentence
	return True if and only if the sentence has a projective dependency tree
	"""
	arcs = {}
	for tok in toks:
		if tok[3] not in arcs:
			arcs[tok[3]] = list()
		arcs[tok[3]].append(tok[0])
	# Implement your code below
	for cur, heads in arcs.items():
		reachable = set()
		stack = [cur]
		while len(stack) != 0:
			q = stack.pop()
			reachable.add(q)
			if q in arcs:
				stack.extend(arcs[q])
		for head in heads:
			for i in range(cur, head + 1):
				if i != 0 and i not in reachable:
					return False
	return True


# Question 2.a.
def perform_shift(wbuffer, stack, arcs,
                  configurations, gold_transitions):
	"""
	perform the SHIFT operation
	"""

	# Implement your code below
	# your code should:
	# 1. append the latest configuration to configurations
	configurations.append((wbuffer.copy(), stack.copy(), arcs.copy()))
	# 2. append the latest action to gold_transitions
	gold_transitions.append("SHIFT")
	# 3. update wbuffer, stack and arcs accordingly
	# hint: note that the order of operations matters
	# as we want to capture the configurations and transition rules
	# before making changes to the stack, wbuffer and arcs
	stack.append(wbuffer.pop())


# Question 2.b.
def perform_arc(direction, dep_label,
                wbuffer, stack, arcs,
				configurations, gold_transitions):
	"""
	params:
		- direction: {"LEFT", "RIGHT"}
		- dep_label: label for the dependency relations
	Perform LEFTARC_ and RIGHTARC_ operations
	"""
	# Implement your code below
	# your code should:
	# 1. append the latest configuration to configurations
	configurations.append((wbuffer.copy(), stack.copy(), arcs.copy()))
	# 2. append the latest action to gold_transitions
	gold_transitions.append("{0}ARC_{1}".format(direction, dep_label))
	# 3. update wbuffer, stack and arcs accordingly
	if direction == "LEFT":
		top = stack.pop()
		second = stack.pop()
		stack.append(top)
		arcs.append((dep_label, top, second))
	else:
		top = stack.pop()
		arcs.append((dep_label, stack[-1], top))
	# hint: note that the order of operations matters
	# as we want to capture the configurations and transition rules
	# before making changes to the stack, wbuffer and arcs


# Question 2
def tree_to_actions(wbuffer, stack, arcs, deps):
	"""
	params:
	wbuffer: a list of word indices, top of buffer is at the end of the list
	stack: a list of word indices, top of buffer is at the end of the list
	arcs: a list of (label, head, dependent) tuples

	Given wbuffer, stack, arcs and deps
	Return configurations and gold_transitions (actions)
	"""

	# configurations:
	# A list of tuples of lists
	# [(wbuffer1, stack1, arcs1), (wbuffer2, stack2, arcs2), ...]
	# Keeps tracks of the states at each step
	configurations=[]

	# gold_transitions:
	# A list of action strings, e.g ["SHIFT", "LEFTARC_nsubj"]
	# Keeps tracks of the actions at each step
	gold_transitions=[]

	# Implement your code below
	# hint:
	# 1. configurations[i] and gold_transitions[i] should
	# correspond to the states of the wbuffer, stack, arcs
	# (before the action was take) and action to take at step i
	# 2. you should call perform_shift and perform_arc in your code
	def is_left_arc():
		return len(stack) >= 2 and \
				stack[-1] in deps and \
				(stack[-1], stack[-2]) in deps[stack[-1]]

	def perform_left_arc():
		top, second = stack[-1], stack[-2]
		tag = deps[top][(top, second)]
		perform_arc("LEFT", tag, wbuffer, stack, arcs, configurations, gold_transitions)

	def is_right_arc():
		baseline = len(stack) >= 2 and \
				stack[-2] in deps and \
				(stack[-2], stack[-1]) in deps[stack[-2]]
		if baseline:
			if stack[-1] in deps:
				for dep in deps[stack[-1]]:
					if dep[1] in wbuffer:
						return False
		return baseline 

	def perform_right_arc():
		top, second = stack[-1], stack[-2]
		tag = deps[second][(second, top)]
		perform_arc("RIGHT", tag, wbuffer, stack, arcs, configurations, gold_transitions)

	def shift():
		perform_shift(wbuffer, stack, arcs, configurations, gold_transitions)

	while len(wbuffer) != 0:
		if is_left_arc():
			perform_left_arc()
		elif is_right_arc():
			perform_right_arc()
		else:
			shift()

	while len(stack) > 1:
		if is_left_arc():
			perform_left_arc()
		else:
			perform_right_arc()

	return configurations, gold_transitions

def isvalid(stack, wbuffer, action):
	"""
	Helper function that returns True only if an action is
	legal given the current states of the stack and wbuffer
	"""
	if action == "SHIFT" and len(wbuffer) > 0:
		return True
	if action.startswith("RIGHTARC") and len(stack) > 1 and stack[-1] != 0:
		return True
	if action.startswith("LEFTARC") and len(stack) > 1 and stack[-2] != 0:
		return True

	return False


# Question 3
def action_to_tree(tree, predictions, wbuffer, stack, arcs):
	"""
	params:
	tree:
	a dictionary of dependency relations (head, dep_label)
		{
			child1: (head1, dep_lebel1),
		 	child2: (head2, dep_label2), ...
		}

	predictions:
	a numpy column vector of probabilities for different dependency labels
	as ordered by the global variable reverse_labels
	predictions.shape = (1, total number of dependency labels)

	wbuffer: a list of word indices, top of buffer is at the end of the list
	stack: a list of word indices, top of buffer is at the end of the list
	arcs: a list of (label, head, dependent) tuples

	"""
	global reverse_labels

	# Implement your code below
	# hint:
	# 1. the predictions contains the probability distribution for all
	# possible actions for a single step, and you should choose one
	# and update the tree only once
	# 2. some actions predicted are not going to be valid
	# (e.g., shifting if nothing is on the buffer)
	# so sort probs and keep going until we find one that is valid.



# ============================================================
# THE FOLLOWING CODE WILL BE PROVIDED
# ============================================================
def get_oracle(toks):
	"""
	Return pairs of configurations + gold transitions (actions)
	from training data
	configuration = a list of tuple of:
		- buffer (top of buffer is at the end of the list)
		- stack (top of buffer is at the end of the list)
		- arcs (a list of (label, head, dependent) tuples)
	gold transitions = a list of actions, e.g. SHIFT
	"""

	stack = [] # stack
	arcs = [] # existing list of arcs
	wbuffer = [] # input buffer

	# deps is a dictionary of head: dependency relations, where
	# dependency relations is a dictionary of the (head, child): label
	# deps = {head1:{
	#               (head1, child1):dependency_label1,
	# 				(head1, child2):dependency_label2
	#              }
	#         head2:{
	#               (head2, child3):dependency_label3,
	# 				(head2, child4):dependency_label4
	#              }
	#         }
	deps = {}

	# ROOT
	stack.append(0)

	# initialize variables
	for position in reversed(toks):
		(idd, _, _, head, lab) = position

		dep = (head, idd)
		if head not in deps:
			deps[head] = {}
		deps[head][dep] = lab

		wbuffer.append(idd)

	# configurations:
	# A list of (wbuffer, stack, arcs)
	# Keeps tracks of the states at each step
	# gold_transitions:
	# A list of action strings ["SHIFT", "LEFTARC_nsubj"]
	# Keeps tracks of the actions at each step
	configurations, gold_transitions = tree_to_actions(wbuffer, stack, arcs, deps)
	return configurations, gold_transitions


# Bonus Question
def featurize_configuration(configuration, tokens, postags):
	"""
	!EXTRA CREDIT!:
	Add new features here to improve the performance of the parser

	Given configurations of the stack, input buffer and arcs,
	words of the sentence and POS tags of the words,
	return some features
	"""
	wbuffer, stack, arcs = configuration
	feats = {}

	feats["%s_%s" % ("len_buffer", len(wbuffer))] = 1
	feats["%s_%s" % ("len_stack", len(stack))] = 1

	# single-word features
	if len(stack) > 0:
		feats["%s_%s" % ("stack_word_1", tokens[stack[-1]])] = 1
		feats["%s_%s" % ("stack_tag_1", postags[stack[-1]])] = 1
		feats["%s_%s_%s" % ("stack_tag_word_1", tokens[stack[-1]], postags[stack[-1]])] = 1

	if len(stack) > 1:
		feats["%s_%s" % ("stack_word_2", tokens[stack[-2]])] = 1
		feats["%s_%s" % ("stack_tag_2", postags[stack[-2]])] = 1
		feats["%s_%s_%s" % ("stack_tag_word_2", tokens[stack[-2]], postags[stack[-2]])] = 1

	if len(wbuffer) > 0:
		feats["%s_%s" % ("buffer_word_1", tokens[wbuffer[-1]])] = 1
		feats["%s_%s" % ("buffer_tag_1", postags[wbuffer[-1]])] = 1
		feats["%s_%s_%s" % ("buffer_tag_word_1", tokens[wbuffer[-1]], postags[wbuffer[-1]])] = 1

	# word-pair features
	if len(stack) > 1:
		feats["%s_%s_%s_%s" % ("stack1_word_tag_stack2_tag", tokens[stack[-1]], postags[stack[-1]], postags[stack[-2]])] = 1

	return feats


def get_oracles(filename):
	"""
	Get configurations, gold_transitions from all sentences
	"""
	with open(filename) as f:
		toks, tokens, postags = [], {}, {}
		tokens[0] = "ROOT"
		postags[0] = "ROOT"

		# a list of all features for each transition step
		feats = []
		# a list of labels, e.g. SHIFT, LEFTARC_DEP_LABEL, RIGHTARC_DEP_LABEL
		labels = []

		for line in f:
			cols = line.rstrip().split("\t")

			if len(cols) < 2: # at the end of each sentence
				if len(toks) > 0:
					if is_projective(toks): # only use projective trees
						# get all configurations and gold standard transitions
						configurations, gold_transitions = get_oracle(toks)

						for i in range(len(configurations)):
							feat = featurize_configuration(configurations[i], tokens, postags)
							label = gold_transitions[i]
							feats.append(feat)
							labels.append(label)

					# reset vars for the next sentence
					toks, tokens, postags = [], {}, {}
					tokens[0] = "ROOT"
					postags[0] = "ROOT"
				continue

			if cols[0].startswith("#"):
				continue

			# construct the tuple for each word in the sentence
			# for each word in the sentence
			# idd: index of a word in a sentence, starting from 1
			# tok: the word itself
			# pos: pos tag for that word
			# head: parent of the dependency
			# lab: dependency relation label
			idd, tok, pos, head, lab = int(cols[0]), cols[1], cols[4], int(cols[6]), cols[7]
			toks.append((idd, tok, pos, head, lab))

			# feature for training to predict the gold transition
			tokens[idd], postags[idd] = tok, pos

		return feats, labels


def train(feats, labels):
	"""
	Train transition-based parsed to predict next action (labels)
	given current configuration (featurized by feats)
	Return the classifier trained using the logistic regression model
	"""
	global feature_vocab, label_vocab, fmax, reverse_labels, reverse_features

	lid = 0 # label ID
	D = len(feats) # each row of feats corresponds to a row in labels
	feature_counts = Counter()

	# build dictionary of labels
	for i in range(D):
		for f in feats[i]:
			feature_counts[f] += 1

		if labels[i] not in label_vocab:
			label_vocab[labels[i]] = lid
			lid += 1

	# build dictionary of features
	for f in feature_counts:
		if feature_counts[f] > MINIMUM_FEAT_COUNT and f not in feature_vocab:
			feature_vocab[f] = fmax
			fmax += 1

	# build reverse lookup for features and labels
	reverse_labels = [None]*lid
	for label in label_vocab:
		reverse_labels[label_vocab[label]] = label

	reverse_features = [None]*fmax
	for feature in feature_vocab:
		reverse_features[feature_vocab[feature]] = feature

	# X is a D-by-fmax matrix, where each row represents
	# features for a configuration / actions
	# Y is a list of labels for all configurations
	X = sparse.lil_matrix((D, fmax))
	Y = []
	for i in range(D):
		for f in feats[i]:
			if f in feature_vocab:
				fid = feature_vocab[f]
				X[i,fid] = 1
		Y.append(label_vocab[labels[i]])

	print ("Docs: ", D, "Features: ", fmax)
	log_reg = linear_model.LogisticRegression(C=L2_REGULARIZATION_STRENGTH,
	                                          penalty='l2')

	clf = grid_search.GridSearchCV(log_reg, {'C':(1e0,1e1)}, n_jobs=10)
	log_reg = clf.fit(sparse.coo_matrix(X), Y).best_estimator_
	print ("Best C: %s" % clf.best_estimator_)

	return log_reg


def parse(toks, logreg):
	"""
	parse sentence with trained model and return correctness measure
	"""
	tokens, postags = {}, {}
	tokens[0] = "ROOT"
	postags[0] = "ROOT"

	heads, labels = {}, {}

	wbuffer, stack, arcs = [], [], []
	stack.append(0)

	for position in reversed(toks):
		# featurization
		(idd, tok, pos, head, lab) = position
		tokens[idd] = tok
		postags[idd] = pos

		# keep track of gold standards for performance evaluation
		heads[idd], labels[idd] = head, lab

		# update buffer
		wbuffer.append(idd)

	tree = {}
	while len(wbuffer) >= 0:
		if len(wbuffer) == 0 and len(stack) == 0: break
		if len(wbuffer) == 0 and len(stack) == 1 and stack[0] == 0:	break

		feats = (featurize_configuration((wbuffer, stack, arcs), tokens, postags))
		X = sparse.lil_matrix((1, fmax))
		for f in feats:
			if f in feature_vocab:
				X[0,feature_vocab[f]]=1

		predictions = logreg.predict_proba(X)
		# your function will be called here
		action_to_tree(tree, predictions, wbuffer, stack, arcs)

	# correct_unlabeled: total number of correct (head, child) dependencies
	# correct_labeled: total number of correctly *labeled* dependencies
	correct_unlabeled, correct_labeled, total = 0, 0, 0

	for child in tree:
		(head, label) = tree[child]
		if head == heads[child]:
			correct_unlabeled += 1
			if label == labels[child]: correct_labeled += 1
		total += 1

	return [correct_unlabeled, correct_labeled, total]


def evaluate(filename, logreg):
	"""
	Evaluate the performance of a parser against gold standard
	"""
	with open(filename) as f:
		toks=[]
		totals = np.zeros(3)
		for line in f:
			cols=line.rstrip().split("\t")

			if len(cols) < 2: # end of a sentence
				if len(toks) > 0:
					if is_projective(toks):
						tots = np.array(parse(toks, logreg))
						totals += tots
						print ("%.3f\t%.3f\t%s" % (totals[0]/totals[2], totals[1]/totals[2], totals))
					toks = []
				continue

			if cols[0].startswith("#"):
				continue

			idd, tok, pos, head, lab = int(cols[0]), cols[1], cols[4], int(cols[6]), cols[7]
			toks.append((idd, tok, pos, head, lab))


def main():
	# this takes a while to train, so can subset the training data
	# (e.g. train.projective.short.conll)
	feats, labels = get_oracles(sys.argv[1])
	logreg = train(feats, labels)
	evaluate(sys.argv[2], logreg)

# to run:
# python hw6.py train.projective.conll dev.projective.conll
if __name__ == "__main__":
    main()
