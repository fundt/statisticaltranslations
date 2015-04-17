import collections, codecs, itertools, re, math

class SMT(object):
	"""docstring for SMT"""
	def match_texts(self, s, t):
		en = codecs.open(s, 'r', 'utf-8')
		es = codecs.open(t, 'r', 'utf-8')
		res = [(re.sub(' ?&apos;', '\'', s).strip().split(), t.strip().split()) \
			for (s, t) in zip(en, es)]
		en.close()
		es.close()
		return res
			
	def init_table(self, en):
		table = collections.defaultdict(lambda: collections.defaultdict(lambda: 0.0))
		for (s, t) in self.texts:
			s = filter(lambda x: x.isalpha() or len(x) > 1, s)
			t = filter(lambda x: x.isalpha() or len(x) > 1, t)
			for w in s:
				for v in t:
					if en:
						table[w][v] += 1
					else:
						table[v][w] += 1
		return table

	def __init__(self, en, es):
		self.texts = self.match_texts(en, es)
		self.en_table = self.init_table(True)
		self.es_table = self.init_table(False)
		self.normalize()
		self.en = en
		self.es = es
		self.iters = 0
		print 'SMT Initalized'

	def EM(self, iters):
		for i in range(iters):
			temp_en_tuples = collections.defaultdict(lambda: 0.0)
			temp_en_counts = collections.defaultdict(lambda: 0.0)
			temp_es_tuples = collections.defaultdict(lambda: 0.0)
			temp_es_counts = collections.defaultdict(lambda: 0.0)
			#E step
			for (e, f) in self.texts:
				e = filter(lambda x: x.isalpha() or len(x) > 1, e)
				f = filter(lambda x: x.isalpha() or len(x) > 1, f)
				for ew in e:
					for fw in f:
						# en to es
						temp_en_tuples[(ew, fw)] += self.en_table[ew][fw]
						temp_en_counts[fw] += self.en_table[ew][fw]
						# es to en
						temp_es_tuples[(fw, ew)] += self.es_table[fw][ew]
						temp_es_counts[ew] += self.es_table[fw][ew]

			self.en_table = collections.defaultdict(lambda: collections.defaultdict(lambda: 0.0))
			self.es_table = collections.defaultdict(lambda: collections.defaultdict(lambda: 0.0))
			for (e, f) in temp_en_tuples:
				self.en_table[e][f] = temp_en_tuples[(e, f)] / temp_en_counts[f]
			for (f, e) in temp_es_tuples:
				self.es_table[f][e] = temp_es_tuples[(f, e)] / temp_es_counts[e]
			#M step
			self.normalize()
			self.iters += 1
			print "EM Iter: ", self.iters

	def indiv_normalize(self, table):
		for ew in table:
			norm_factor = sum(table[ew][fw] for fw in table[ew])
			if norm_factor != 0:
				for fw in table[ew]:
					table[ew][fw] /= norm_factor

	def normalize(self):
		# en to es
		self.indiv_normalize(self.en_table)
		# es to en
		self.indiv_normalize(self.es_table)

	def es2en_alignment(self, e, f):
		res = []
		for i in range(len(f)):
			max_p = 0.0
			max_i = None
			for j in range(len(e)):
				if self.es_table[f[i]][e[j]] > max_p:
					max_p = self.es_table[f[i]][e[j]]
					max_i = j
			res.append((max_i, i))
		return res

	def en2es_alignment(self, e, f):
		res = []
		for i in range(len(e)):
			max_p = 0.0
			max_i = None
			for j in range(len(f)):
				if self.en_table[e[i]][f[j]] > max_p:
					max_p = self.en_table[e[i]][f[j]]
					max_i = j
			res.append((i, max_i))
		return res

	def neighbor_in_matrix(self, entry, intersection):
		if entry[0] == None or entry[1] == None:
			return False
		h_neighbors = [(entry[0] + 1, entry[1]), (entry[0] - 1, entry[1])]
		v_neighbors = [(entry[0], entry[1] + 1), (entry[0], entry[1] - 1)]

		has_H = False
		for n in h_neighbors:
			if n in intersection:
				has_H = True
		has_V = False
		for n in v_neighbors:
			if n in intersection:
				has_V = True
		
		if has_H and has_V:
			return False
		return has_H or has_V

	def in_matrix(self, e, f, entry, intersection):
		i, j = entry
		max_i = len(e)
		max_j = len(f)
		for x in range(max_j):
			if (i, x) in intersection:
				return True
		for x in range(max_i):
			if (x, j) in intersection:
				return True
		return False
 
	def intersection_matrix(self, e, f):
		# returns the intersection matrix between e and f
		# in the format of a set of tuples (i, j), where i
		# is the index of the e_word and j is the index of 
		# the f_word in the alignment. This allows for 
		# each e_word to map to multiple values of j
		# and for each f_word to map to multiple values of i
		es2en = set(self.es2en_alignment(e, f))
		en2es = set(self.en2es_alignment(e, f))
		intersection = es2en.intersection(en2es)

		for entry in es2en.union(en2es).difference(intersection):			
			if entry in es2en or entry in en2es: 
				if not self.in_matrix(e, f, entry, intersection) or \
				  		self.neighbor_in_matrix(entry, intersection):
					intersection.update([entry])
		return sorted(intersection)


	def PTT(self):
		"""
		Returns the phrase translation probability table in the form of:
		pt[es_phrase][en_phrase] = probability
		"""
		phrase_table = collections.defaultdict(lambda: collections.Counter())
		ptable = collections.defaultdict(lambda: collections.defaultdict(lambda: 0.0))
		for (e, f) in self.texts:
			e = filter(lambda x: x.isalpha() or len(x) > 1, e)
			f = filter(lambda x: x.isalpha() or len(x) > 1, f)
			im = self.intersection_matrix(e, f)
			es_phrases = collections.defaultdict(lambda: [])
			en_phrases = collections.defaultdict(lambda: [])
			for j in range(len(f)):
				for i in range(len(e)):
					if (i,j) in im:
						es_phrases[i].append(j)
						en_phrases[j].append(i)
			# get consistent phrase pairs
			for j in range(len(f)):
				if j not in en_phrases:
					continue
				for i in range(j, len(f)):
					span = sorted(set(range(j, i + 1)))
					D = sorted(set(itertools.chain(*map(lambda x: en_phrases[x], span))))
					R = sorted(set(list(itertools.chain(*map(lambda x: es_phrases[x], D)))))
					if R == span:
						e_phrase = ' '.join([e[x] for x in sorted(D)])
						f_phrase = ' '.join([f[x] for x in sorted(R)])
						ptable[e_phrase][f_phrase] += 1

		for e_phrase in ptable:
			evals = sum(ptable[e_phrase].values())
			if evals > 0:
				for f_phrase in ptable[e_phrase]:
					phrase_table[f_phrase][e_phrase] = ptable[e_phrase][f_phrase]*len(ptable[e_phrase])*1.0/evals
		print 'PTT Initalized'
		return phrase_table
				


	def M1(self):
		ttable = collections.defaultdict(lambda: collections.Counter())
		self.ttable = collections.defaultdict(lambda: collections.Counter())
		for (e, f) in self.texts:
			e = filter(lambda x: x.isalpha() or len(x) > 1, e)
			f = filter(lambda x: x.isalpha() or len(x) > 1, f)
			m = collections.defaultdict(lambda: [])
			for i, j in self.es2en_alignment(e, f):
				if i == None:
					continue
				m[j].append(i)
			for j in m:
				ttable[' '.join(e[x] for x in sorted(m[j]))][f[j]] += 1

		for j in ttable:
			norm = sum(ttable[j].values())
			if norm > 0:
				for i in ttable[j]:
					self.ttable[i][j] = ttable[j][i]*len(ttable[j])*1.0/norm
		print 'M1 Initalized'
		return self.ttable
