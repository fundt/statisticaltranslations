import math
import m1 
import pdb
import pickle
import sys
from collections import defaultdict
from collections import Counter
import itertools
import codecs
import re

# add trui/bi constants to increase effectiveness in flip
# lowercase after dot in phrase table
# check false in generate hypothesis
# Length punishment W favors one-to-manny
# Skipm punish divide if we do NULL

# favor many-to-many

# A - an changes

BEAM_SIZE = 10
WORD_SKIP_LIMIT = 2
MAX_PHRASE_LEN = 3
CONSTANT = 2
SKIP_PUNISH = 300000
W = 2
translation_power = 1
lm_power = 3
num_translations = 10
UNIGRAM_FILENAME = "w1.txt"
BIGRAM_FILENAME = "w2.txt"
TRIGRAM_FILENAME = "w3.txt"
debug = "s"
testing = False

class PriorityStack():
	def __init__(self, ttable, language_model, max_phrase_len, beam_size):
		self.ttable = ttable
		self.states = []
		self.max_phrase_len = max_phrase_len
		self.beam_size = beam_size
		self.lm = language_model

	def get_states(self):
		return self.states

	def get_lang_prob(self, sentence, index):
		total = self.lm["TOTAL_BI"]
		uni_total = self.lm["TOTAL_UNI"]
		tri_total = self.lm["TOTAL_TRI"]
		log_sum = -1
		# print self.lm;
		# print "Starting to analyze LM Probs for ", sentence
		for i in range(index, len(sentence)):
			if (sentence[i]) not in self.lm:
				# print sentence[i], "NOT in the LM minus ", math.log(float(uni_total)) 
				log_sum -= math.log(float(uni_total))
			else:
				log_sum += (math.log((self.lm[sentence[i]]+1)/float(uni_total)))
				# print "Log sum plus UNIGRAM: ", math.log((self.lm[sentence[i]]+1)/float(uni_total))
			if (i > 0):
				if (sentence[i-1], sentence[i]) not in self.lm:
					log_sum -= (math.log(float(total)))
					# print (sentence[i-1], sentence[i]), "NOT in the LM minus ", math.log(float(total)) 
					continue
				elif (sentence[i-1]) in self.lm:
					log_sum += 0.5*(math.log((self.lm[(sentence[i-1], sentence[i])] + 1)/float(self.lm[sentence[i-1]]+uni_total)))
					# print sentence[i], "IN LM Log sum plus BIGRAM: ", math.log((self.lm[(sentence[i-1], sentence[i])] + 1)/float(total))
				else:
					log_sum += 0.25*(math.log((self.lm[(sentence[i-1], sentence[i])] + 1)/float(total)))
					# print sentence[i], "NOT! IN LM Log sum plus BIGRAM: ", (math.log((self.lm[(sentence[i-1], sentence[i])] + 1)/float(uni_total*2)))
			if (i > 1):
				if (sentence[i-2], sentence[i-1], sentence[i]) not in self.lm:
					# print (sentence[i-2], sentence[i-1], sentence[i]), "NOT in the LM minus ", math.log(float(tri_total))
					log_sum -= 0.5*math.log(float(tri_total))
					continue
				elif (sentence[i-2]) in self.lm and (sentence[i-1]) in self.lm:
					log_sum += 0.5*(math.log((self.lm[(sentence[i-2], sentence[i-1], sentence[i])] + 1)/float(self.lm[sentence[i-2]]+self.lm[sentence[i-1]]+uni_total)))
				# else:
					# log_sum += 0.25*(math.log((self.lm[(sentence[i-2], sentence[i-1], sentence[i])] + 1)/float(tri_total)))
				# print "Log sum plus TRIGRAM: ", (math.log((self.lm[(sentence[i-2], sentence[i-1], sentence[i])] + 1)/float(tri_total)))

		# return  1
		# print "LM RETURNS: ", math.pow(-1.0/(log_sum), lm_power)
		return math.pow(-1.0/(log_sum), lm_power)

	def get_d(self, prev_eng, eng_phrase, prev_esp, esp_phrase):
		return 1 # remember to get the d 


	def get_english_phrases(self, prev_state, esp_phrase, stack_no):
		prev_score = prev_state[0	]
		candidate_phrases = [(b,a) for a,b  in self.ttable[" ".join(esp_phrase)].iteritems()]
		if " ".join(esp_phrase) == "las" or " ".join(esp_phrase) == "las" or " ".join(esp_phrase) == "la" or " ".join(esp_phrase) == "el":
			candidate_phrases = [(100, "the")] 
		elif " ".join(esp_phrase) == "Las" or " ".join(esp_phrase) == "Las" or " ".join(esp_phrase) == "La" or " ".join(esp_phrase) == "El":
			candidate_phrases = [(100, "The")] 	
		if len(candidate_phrases) == 0 and len(esp_phrase) == 1 and " ".join(esp_phrase) != " ".join(esp_phrase).lower():
			if debug != "f": print "PANIC, no find"
			candidate_phrases = [(100, esp_phrase[0])]	
		if (len(esp_phrase)==1) and (" ".join(esp_phrase).isdigit() or " ".join(esp_phrase)=="%"):
			candidate_phrases = [(100, esp_phrase[0])]	
		if (len(esp_phrase)==1) and " ".join(esp_phrase) == "&quot;":
			candidate_phrases = [(100, esp_phrase[0])]


		candidate_phrases = filter(lambda x: len(x[1].split())<5, candidate_phrases)
		phis = [x[0] for x in reversed(sorted(candidate_phrases))][:num_translations]
		candidate_phrases = [x[1].split() for x in reversed(sorted(candidate_phrases))][:num_translations]
		possible_phrases = []

		for n, eng_phrase in enumerate(candidate_phrases):
			new_index = len(prev_state[2])
			new_hypothesis = prev_state[2] + eng_phrase
			length_factor = math.pow(W, (len(eng_phrase)))

			score = prev_score 	* math.pow(phis[n], len(esp_phrase)) * self.get_lang_prob(new_hypothesis, new_index) * length_factor
			if (len(prev_state[1]) + len(esp_phrase) < stack_no): score = score / math.pow(SKIP_PUNISH, stack_no - (len(prev_state[1]) + len(esp_phrase)))
			# score *= self.get_d(prev_state[2], eng_phrase, prev_state[1], esp_phrase)
			possible_phrases.append((eng_phrase, score, (prev_score, math.pow(phis[n], len(esp_phrase)), self.get_lang_prob(new_hypothesis, new_index), length_factor)))

		return possible_phrases #tuple of (eng_phrase, score)


	def add(self, prev_state, esp_phrase, stack_no):
		possible_phrases = self.get_english_phrases(prev_state, esp_phrase, stack_no)
		global debug
		if debug != 'f': print "possible_phrases:", possible_phrases
		for eng_phrase, score, breakdown in possible_phrases:
			if debug == "s":
				a = raw_input()
				if a == "c":
					debug = "c"
				elif a == "f":
					debug = "f"

			if len(self.states) < self.beam_size:
				self.states.append((score, prev_state[1] + esp_phrase, prev_state[2] + eng_phrase, stack_no))
				if debug != 'f':
					print("Added:", prev_state[1], esp_phrase)
					print("   English:", prev_state[2], eng_phrase)
					print("   Score:", score)
					print("   prev:", breakdown[0])
					print("   trans:", breakdown[1])
					print("   lang_model:", breakdown[2])
					print("   length:", breakdown[3])
				self.states = list(reversed(sorted(self.states)))
			elif score > self.states[-1][0]:
				self.states.pop()
				self.states.append((score, prev_state[1] + esp_phrase, prev_state[2] + eng_phrase, stack_no))
				self.states = list(reversed(sorted(self.states)))
				if debug != 'f':
					print("Added:", prev_state[1], esp_phrase)
					print("   English:", prev_state[2], eng_phrase)
					print("   Score:" , score)
					print("   prev:", breakdown[0])
					print("   trans:", breakdown[1])
					print("   lang_model:", breakdown[2])
					print("   length:", breakdown[3])

def generate_hypotheses(stacks, esp_s, state):
	global iters
	esp_start = state[3]
	for i in range(esp_start, min(esp_start + WORD_SKIP_LIMIT, len(esp_s))):
		for phrase_len in range(1, min(MAX_PHRASE_LEN + 1, len(esp_s) - i + 1)):
			new_esp_phrase = esp_s[i : i+phrase_len]
			new_esp_phrase_lower = [a.lower() for a in esp_s[i : i+phrase_len]]
			if debug != "f":
				print("Trying to add:", new_esp_phrase)
			stacks[i + phrase_len].add(state, new_esp_phrase, i + phrase_len)
			if new_esp_phrase != new_esp_phrase_lower:	
				stacks[i + phrase_len].add(state, new_esp_phrase_lower, i + phrase_len)

def decode(esp_s, ttable, lm):
	sentence_len = len(esp_s)
	stacks = [PriorityStack(ttable, lm, MAX_PHRASE_LEN, BEAM_SIZE) for n in range(sentence_len + 1)]
	stacks[0].states.append([1, [], [], 0])
	global debug

	for stack in stacks:
		if debug != "f":
			for n, s in enumerate(stacks):
				print "STACK:", n
				for st in s.get_states():
					print st

		if debug == "c":
			a = raw_input()
			if a == "s" or a == "f":
				debug = a
		for state in stack.get_states():
			if (debug != 'f'):
				print("\n\n\n")
				print("Generating hypotheses...")
				print("State:", state)

			generate_hypotheses(stacks, esp_s, state)

	for stack in reversed(stacks):
		if stack.get_states():
			global testing
			if testing:
				print(" --- ".join([str(a) for a in stack.get_states()[0]]))
			return " ".join(stack.get_states()[0][2])


def generate_language_model(unigram_filename, bigram_filename, trigram_filename):
	lm = {}
	total_uni_count = 0
	total_count = 0
	total_tri_count = 0
	for line in open(unigram_filename, "r"):
		try:
			word, count = line.split("\t")
			count = int(count)
			total_uni_count += count
			lm[word] = count
		except:
			pass
	print "unigram done"

	for line in open(bigram_filename, "r"):
		try:
			count, w1, w2 = line.split()
			count = int(count)
			total_count += count
			lm[(w1,w2)] = count
		except:
			pass
	print "bigram done"

	for line in open(trigram_filename, "r"):
		try:
			count, w1, w2, w3 = line.split()
			count = int(count)
			total_tri_count += count
			lm[(w1,w2,w3)] = count
		except:
			pass
	lm["TOTAL_UNI"] = total_uni_count
	lm["TOTAL_BI"] = total_count
	lm["TOTAL_TRI"] = total_tri_count
	# print "LM _________________________________"
	# print lm
	return lm


def get_lang_prob(lm, sentence):
	total = lm["TOTAL_BI"]
	uni_total = lm["TOTAL_UNI"]
	tri_total = lm["TOTAL_TRI"]
	log_sum = -1
	for i in range(0, len(sentence)):
		if (sentence[i]) not in lm:
			log_sum -= math.log(float(uni_total))
		else:
			log_sum += math.log((lm[sentence[i]]+1)/float(uni_total))

		if (i > 0):
			if (sentence[i-1], sentence[i]) not in lm:
				log_sum -= math.log(float(total))
				continue
			log_sum += math.log((lm[(sentence[i-1], sentence[i])] + 1)/float(total))
		if (i > 1):
			if (sentence[i-2], sentence[i-1], sentence[i]) not in lm:
				log_sum -= math.log(float(tri_total))
				continue
			log_sum += math.log((lm[(sentence[i-2], sentence[i-1], sentence[i])] + 1)/float(total))

	return math.pow(-1.0/(log_sum), lm_power)



def mean(arr):
	return sum(arr)/float(len(arr))

def compute_sentence(es_sentence, ttable, lm):
	new_sentence = []
	for word in es_sentence:
		candidate_phrases = [(b,a) for a,b  in ttable[word].iteritems()]
		candidate_phrases = filter(lambda x: len(x[1].split())<5, candidate_phrases)
		phis = [x[0] for x in reversed(sorted(candidate_phrases))][:num_translations]
		candidate_phrases = [x[1].split() for x in reversed(sorted(candidate_phrases))][:num_translations]
		best_score = -1
		best_word = ""
		for n, curr in enumerate(candidate_phrases):
			score =  mean([get_lang_prob(lm, curr) for word in curr]) * math.pow(phis[n], translation_power)
			print curr,	 "Score:", score, " lm", mean([get_lang_prob(lm, curr) for word in curr]), "trans", math.pow(phis[n], translation_power)
			if score > best_score:
				best_score = score
				best_word = curr
		print "adding word:", " ".join(best_word)
		new_sentence.append(" ".join(best_word))

	print "english sentence:", new_sentence
	#now we have our most likely set of new_words

	for i in range(0, len(new_sentence)):
		possibilities = itertools.permutations(new_sentence[i:i+CONSTANT])
		best_phrase = []
		best_score = -1
		for possibility in possibilities:
			curr_score = get_lang_prob(lm, possibility)
			if curr_score > best_score:
				best_phrase = possibility
				best_score = curr_score

		new_sentence[i:i+CONSTANT] = best_phrase
	print "\n\nTranslation:", new_sentence
	return " ".join(new_sentence)

def split_punctuation(sentence):
	#return re.split(r'[,;\.!:?]+', sentence)	
	return sentence.split(" , ") #TODO FIX

def shuffle(lm, sentence):
	new_sentence = sentence.split()
	for i in range(0, len(new_sentence)):
		possibilities = itertools.permutations(new_sentence[i:i+CONSTANT])
		best_phrase = []
		best_score = -1
		for possibility in possibilities:
			curr_score = get_lang_prob(lm, possibility)
			if curr_score > best_score:
				best_phrase = possibility
				best_score = curr_score

		new_sentence[i:i+CONSTANT] = best_phrase
	print "\n\nTranslation:", new_sentence
	return " ".join(new_sentence)

def main():
	global debug
	iters = 20
	if '-c' in sys.argv:
		smt = m1.SMT('../es-en/train/europarl-v7.es-en.en', '../es-en/train/europarl-v7.es-en.es')
		print('SMT Initialized')
		smt.EM(iters) # could be more or lesS
		ttable = smt.PTT()
		pickle.dump(dict(ttable), codecs.open("save" + str(iters) + "ItersN.p", "w", "utf-8"))

	ttable = defaultdict(lambda: Counter(), pickle.load(open("save"+str(iters)+"ItersN.p", "rb"))	)

	lm = generate_language_model(UNIGRAM_FILENAME, BIGRAM_FILENAME, TRIGRAM_FILENAME)

	if '-l' in sys.argv:
		while True:
			query = raw_input("Sentence: ")
			print("Score: ", get_lang_prob(lm, query.split()))

	if "-p" in sys.argv:
		eng_sentences = codecs.open("../es-en/test/" + sys.argv[3], "w", "utf-8")
		esp_sentences = codecs.open(sys.argv[2], "r", "utf-8").readlines()
		for sentence in esp_sentences:
			sentence = " ".join(filter(lambda x: x!="." and x!="!" and x!="?" and x!=":" and x!=";", sentence.split()))
			parts = split_punctuation(sentence)
			debug = "f"
			#print("Translating", sentence)
			result = " , ".join([shuffle(lm, decode(part.split(), ttable, lm)) for part in parts])
			result += " ."
			#pass#print "\n\n"
			#pass#print "spanish:", sentence
			#pass#print "result", result
			eng_sentences.write(result + '\n')
		eng_sentences.close()
	if "-s" in sys.argv:
		#while True:
		#	esp_s = raw_input("Enter sentence: ")
		#	compute_sentence(esp_s.split(), ttable, lm)
		eng_sentences = codecs.open("../es-en/test/output.en", "w", "utf-8")
		esp_sentences = codecs.open(sys.argv[2], "r","utf-8").readlines()
		for sentence in esp_sentences:
			debug = "s"
			result = compute_sentence(sentence.split(), ttable, lm)
			print result
			#pass#print "\n\n"
			#pass#print "spanish:", sentence
			#pass#print "result", result
			eng_sentences.write(result.encode('utf-8') + '\n')
		eng_sentences.close()

	else:
		while True:
			global testing
			testing = True
			debug = "s"
			esp_s = raw_input("Enter sentence to translate:") 
			esp_s = " ".join(filter(lambda x: x!="." and x!="," and x!="!" and x!="?" and x!=":" and x!=";", esp_s.split()))
			print "FINAL:", shuffle(lm, decode(esp_s.split(), ttable, lm))


if __name__ == "__main__":
	main()