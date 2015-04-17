import shelve,m1 # stack_decoder

def main():
	# train
	smt = m1.SMT('../es-en/train/europarl-v7.es-en.en', '../es-en/train/europarl-v7.es-en.es')
	smt.EM(10) 
	ttable = smt.PTT()
<<<<<<< HEAD


	print(ttable)
	# dev
	"""
	with open('output0', 'w') as output:
		for (e, f) in smt.match_texts('newstest2012.en', 'newstest2012.es')
			output.write(stack_decoder.decode(f, ttable))
	"""	

=======
	while True:
		line = raw_input('key: ')
		if line == '':
			break
		print ttable[line]
>>>>>>> origin/master
if __name__=='__main__':
	main()