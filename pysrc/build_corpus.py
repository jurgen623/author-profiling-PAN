import os
import gensim
import numpy
import sys
from gensim import utils
import logging
from pprint import pprint as pprint
from author_truths import author_truth_dict
from arktwokenizepy import twokenize
from urllib.parse import urlparse, urlunparse
from subprocess import call
from operator import itemgetter
from mystopwords import my_stopwords

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('gensim.models.wrappers.ldamallet')

# training_data_dir = os.path.join(os.curdir, "gensim", "pan14txt")

# training_data_dir = os.path.join(os.curdir, "gensim", "pan14txt", "UNIQUE")
# training_data_dir = os.path.join(os.curdir, "gensim", "en_fewertweets", "Tworpus")
# training_data_dir = os.path.join(os.curdir, "gensim", "Tworpus-OneDrive-2015-05-13", "unique-picky")
training_data_dir = "C:\\Users\\novatech\\topic-models\\Nov-4-2015"

# test_data_dir = os.path.join(os.curdir, "gensim", "pan15txt", "UNIQUE")
test_data_dir = os.path.join(os.curdir, "gensim", "pan15txt", "UNIQUE")
original_training_docs = {"en": [], "es": [], "nl": [], "it": [], "all": []} # ONLY FOR DEVELOPMENT

def my_harsh_tokenize(text):
	words = []
	for word in text.lower().strip().split(" "):
		if not any([ch.isalnum() for ch in word]):
			continue
		word = word.replace("‘", "'").replace("’", "'")
		word = word.strip('\'" :?()\\/,.[]{}!;')
		if len(word) < 2:
			continue
		words.append(word)
	return words


def trim_url(raw_url):
	""" Returns just the scheme and network location part of a url (must begin with a valid scheme like http), for example:
	'http://t.co/v9E09cqCKn' => 't.co'
	'http://4sq.com/5w0x9q' => '4sq.com'
	'4sq.com/5w0x9q' => ''
	'www.google.com' => ''
	"""
	parsed_url = urlparse(raw_url)
	return urlunparse((parsed_url.scheme, parsed_url.netloc, '', '', '', ''))


def get_sim(training_corpus, tfidf_model, latent_model, latent_index, raw_query):
	vec_bow = tfidf_model[training_corpus.dictionary.doc2bow(raw_query.lower().split())]
	vec_lsi = latent_model[vec_bow] # convert the query to LSI space
	print(vec_lsi)
	pprint(gensim.utils.toptexts(vec_lsi, original_training_docs[training_corpus.language], latent_index))
	return

def my_preprocess(doctext):
	words = []
	for word in twokenize.tokenize(doctext.lower()):
		if len(word) < 2 or word.startswith("item_"): # some app seems to post flickr items with item_(sequence of digits)
			continue
		if word.startswith("@"):
			word = "@username"

		word = word.replace("‘", "'").replace("’", "'").replace("…", "...")

		if word.startswith("http"):
			word = trim_url(word)
		elif ".com" in word:
			word = trim_url("http://" + word)
		words.append(word)
	return words

class MyTestCorpus(object):
	""" Collects preprocessed tweets from text files, one file per author, one tweet (document) per line
	"""
	def __init__(self, datadir, dictionary, prefix="en_", extension=".txt", language="en"):
		self.filenames = []
		self.prefix = prefix
		self.extension = extension
		self.language = language
		for root, dirs, files in os.walk(datadir):
			self.filenames.extend( [os.path.join(root, file) for file in files
				 if file.startswith(prefix) and file.endswith(extension) and file != "truth.txt"] )
		print("\nFound", len(self.filenames), extension, "files in", datadir)

		self.filenames = sorted(self.filenames)
		self.files_processed = 0
		self.labels = []

		self.dictionary = dictionary # use an already-built dictionary passed in to the constructor

	def __iter__(self):
		print("  (now iterating over test corpus documents)")

		# if self.files_processed == 0:
		# 	update_dictionary = True
		# 	print("   (first time processing documents: will update corpus dictionary with new words)")
		# else:
		# 	update_dictionary = False

		for filename in self.filenames:
			userid = os.path.basename(filename).replace(self.prefix, "").replace(self.extension, "")
			label = ""
			if userid in author_truth_dict:
				if author_truth_dict[userid]['gender'] is "M":
					gender = 1
				else:
					gender = 0
				label = gender

			for line in open(filename, mode='r', encoding='utf8'):
				# Not sure if we want this constraint in the test corpus or not
				# if len(line) < 30 or len(line.split(" ")) < 5:
				# 	continue

				self.files_processed += 1
				if label is not "":
					self.labels.append(label)

				words = []
				for word in twokenize.tokenize(line.lower()):
					if len(word) < 2 or word.startswith("item_"): # some app seems to post flickr items with item_(sequence of digits)
						continue
					if word.startswith("@"):
						word = "@username"

					word = word.replace("‘", "'").replace("’", "'").replace("…", "...")

					if word.startswith("http"):
						word = trim_url(word)
					elif ".com" in word:
						word = trim_url("http://" + word)
					words.append(word)

				# assume there's one document per line, tokens separated by whitespace
				yield self.dictionary.doc2bow(words)

		print(self.dictionary)

	def __len__(self):
		return self.files_processed

	def get_dictionary(self):
		# print(self.files_processed, "tweets processed so far.", self.dictionary)
		if self.files_processed == 0:
			for document in self:
				pass # force the object to iterate over all its files at least once
		# now the dictionary should no longer be empty
		# print(self.files_processed, "tweets processed so far.", self.dictionary)
		return self.dictionary


class MyTrainingCorpus(object):
	""" Collects preprocessed tweets from text files, one file per author, one tweet (document) per line
	"""
	def __init__(self, datadir, prefix="en_", extension=".txt", language="en"):
		self.filenames = []
		self.prefix = prefix
		self.extension = extension
		self.language = language
		for root, dirs, files in os.walk(datadir):
			self.filenames.extend( [os.path.join(root, file) for file in files
				 if file.startswith(prefix) and file.endswith(extension) and file != "truth.txt"] )
		print("\nFound", len(self.filenames), extension, "files in", datadir)

		self.filenames = sorted(self.filenames)
		self.files_processed = 0
		self.labels = []

		self.dictionary = gensim.corpora.Dictionary()

	def __iter__(self):
		print("  (now iterating over corpus documents)")

		if self.files_processed == 0:
			update_dictionary = True
			print("   (first time processing documents: will update corpus dictionary with new words)")
		else:
			update_dictionary = False

		for filename in self.filenames:
			userid = os.path.basename(filename).replace(self.prefix, "").replace(self.extension, "")
			label = ""
			if userid in author_truth_dict:
				if author_truth_dict[userid]['gender'] is "M":
					gender = 1
				else:
					gender = 0
				label = gender

			for line in open(filename, mode='r', encoding='utf8'):
				if len(line) < 30 or len(line.split(" ")) < 5:
					continue

				self.files_processed += 1
				if label is not "":
					self.labels.append(label)

				words = [] #[word for word in twokenize.tokenize(line.lower()) if len(word)>1]
				for word in twokenize.tokenize(line.lower()):
					if len(word) < 2 or word.startswith("item_"): # some app seems to post flickr items with item_(sequence of digits)
						continue
					if word.startswith("@"):
						word = "@username"

					word = word.replace("‘", "'").replace("’", "'").replace("…", "...")

					if word.startswith("http"):
						word = trim_url(word)
					elif ".com" in word:
						word = trim_url("http://" + word)
					words.append(word)

				# ONLY FOR DEBUGGING:
				if update_dictionary:
					original_training_docs[self.language].append({"fulltext": line, "tokenized": " ".join(words)})

				# assume there's one document per line, tokens separated by whitespace
				yield self.dictionary.doc2bow(words, allow_update=update_dictionary)

		print(self.dictionary)

	def __len__(self):
		return self.files_processed

	def get_dictionary(self):
		# print(self.files_processed, "tweets processed so far.", self.dictionary)
		if self.files_processed == 0:
			for document in self:
				pass # force the object to iterate over all its files at least once
		# now the dictionary should no longer be empty
		# print(self.files_processed, "tweets processed so far.", self.dictionary)
		return self.dictionary

def build_training_corpus(language):
	print("\n-- Begin building {} training corpus --\n".format(language))
	textfiledir = training_data_dir
	# for language in ['en', 'es', 'fr', 'it', 'nl']:
	training_corpus = MyTrainingCorpus(textfiledir, "{}_".format(language), ".txt", language)
	training_corpus.get_dictionary()

	training_corpus.dictionary.save_as_text(os.path.join(textfiledir, "train_dict_{}_noncompact.txt".format(language)), sort_by_word=False)

	# remove stop words (or not) and words that appear only once
	# once_ids = [tokenid for tokenid, docfreq in iter(training_corpus.dictionary.dfs.items()) if docfreq == 1]
	# print("Removing dictionary ids for", len(once_ids), "single-use words")
	# training_corpus.dictionary.filter_tokens(once_ids)

	if language == "all":
		stoplist = my_stopwords['en'] + my_stopwords['es'] + my_stopwords['it'] + my_stopwords['nl']
	else:
		stoplist = my_stopwords[language]
	stop_ids = [training_corpus.dictionary.token2id[stopword] for stopword in stoplist if stopword in training_corpus.dictionary.token2id]
	print("\nremoving dictionary ids for", len(stop_ids), "stop words")
	training_corpus.dictionary.filter_tokens(stop_ids)
	# training_corpus.dictionary.compactify() # remove gaps in id sequence after words that were removed

	training_corpus.dictionary.filter_extremes(no_below=10,no_above=0.4)
	print(training_corpus.dictionary)

	# Just for diagnostic purposes:
	training_corpus.dictionary.save_as_text(os.path.join(textfiledir, "train_dict_{}_compact.txt".format(language)), sort_by_word=False)

	dictionary_filename = os.path.join(textfiledir, "train_gensim_dictionary_{}.dict".format(language))
	corpus_filename = os.path.join(textfiledir, "train_gensim_corpus_{}".format(language))

	print("\nSaving training dictionary to file:", dictionary_filename)
	# training_corpus.dictionary.save(dictionary_filename, separately=None)
	training_corpus.dictionary.save(dictionary_filename)

	print("Saving training corpus to svmlight file:", corpus_filename + ".svmlight")
	#gensim.corpora.SvmLightCorpus.serialize(corpus_filename + ".svmlight", training_corpus, None, None, None, training_corpus.labels)
	gensim.corpora.SvmLightCorpus.serialize(corpus_filename + ".svmlight", training_corpus, None, None, None, None) # we're using unlabeled training data

	print("Saving training corpus to mmcorpus file:", corpus_filename + ".mm")
	gensim.corpora.MmCorpus.serialize(corpus_filename + ".mm", training_corpus)
	print("\n-- Done building {} training corpus --\n".format(language))
	return training_corpus

def build_test_corpus(language, training_corpus, training_dictionary_filename="", training_corpus_filename=""):
	textfiledir = test_data_dir
	print("\n-- Begin building {} test corpus --\n".format(language))
	# for language in ['en', 'es', 'it', 'nl']:
	# training_corpus = gensim.corpora.SvmLightCorpus(training_corpus_filename)
	# training_dictionary = gensim.corpora.Dictionary.load(training_dictionary_filename, mmap=None)

	test_corpus = MyTestCorpus(textfiledir, training_corpus.dictionary, "{}_".format(language), ".txt", language)
	test_corpus.get_dictionary()
	print("Test dictionary:", test_corpus.dictionary)
	print("Test corpus:", test_corpus)

	# Just for diagnostic purposes (to see if tokens and ids match up with the one from the training set only):
	test_corpus.dictionary.save_as_text(os.path.join(textfiledir, "test_dict_{}_compact.txt".format(language)), sort_by_word=False)

	corpus_filename = os.path.join(textfiledir, "test_gensim_corpus_{}".format(language))

	print("Saving test corpus to file:", corpus_filename)
	gensim.corpora.SvmLightCorpus.serialize(corpus_filename + ".svmlight", test_corpus, None, None, None, test_corpus.labels)

	print("\n-- Done building {} test corpus --\n".format(language))
	return test_corpus

class MyLdaMallet(gensim.models.wrappers.LdaMallet):
	def get_tmp_prefix(self):
		# if self.prefix is ".\\gensim\\pan14txt\\UNIQUE\\mallet_files\\mallet_en_", return ".\\gensim\\pan14txt\\UNIQUE\\mallet_files\\tmp\\mallet_en_"
		return os.path.join(os.path.dirname(self.prefix), "tmp", os.path.basename(self.prefix))

	def finferencer(self, infer=False):
		# if infer:
		# 	return self.get_tmp_prefix() + 'inferencer.mallet'
		# else:
		# 	return self.prefix + 'inferencer.mallet'
		# actually, we always want the inferencer from the main (training) directory:
		return self.prefix + 'inferencer.mallet'

	def ftopickeys(self, infer=False):
		if infer:
			return self.get_tmp_prefix() + 'topickeys.txt'
		else:
			return self.prefix + 'topickeys.txt'

	def fstate(self, infer=False):
		if infer:
			return self.get_tmp_prefix() + 'state.mallet.gz'
		else:
			return self.prefix + 'state.mallet.gz'

	def fdoctopics(self, infer=False):
		if infer:
			return self.get_tmp_prefix() + 'doctopics.txt'
		else:
			return self.prefix + 'doctopics.txt'

	def fcorpustxt(self, infer=False):
		if infer:
			return self.get_tmp_prefix() + 'corpus.txt'
		else:
			return self.prefix + 'corpus.txt'

	def fcorpusmallet(self, infer=False):
		if infer:
			return self.get_tmp_prefix() + 'corpus.mallet'
		else:
			return self.prefix + 'corpus.mallet'

	def fwordweights(self, infer=False):
		if infer:
			return self.get_tmp_prefix() + 'wordweights.txt'
		else:
			return self.prefix + 'wordweights.txt'

	def train(self, corpus):
		self.convert_input(corpus, infer=False)
		if not all([os.path.exists(file) for file in [self.fcorpusmallet(), self.fstate(), self.fdoctopics(), self.ftopickeys(), self.finferencer()]]):
			cmd = self.mallet_path + " train-topics --input %s --num-topics %s --alpha 0.5 --optimize-interval %s "\
				"--num-threads %s --output-state %s --output-doc-topics %s --output-topic-keys %s "\
				"--num-iterations %s --inferencer-filename %s --show-topics-interval 500 --xml-topic-report %s --xml-topic-phrase-report %s"
			cmd = cmd % (self.fcorpusmallet(), self.num_topics, 50, self.workers,
						 self.fstate(), self.fdoctopics(), self.ftopickeys(),
						 1000, #self.iterations,
						 self.finferencer(),
						 self.fdoctopics().replace(".txt", ".xml"),
						 self.fdoctopics().replace(".txt", "_phrases.xml")
			)
			# NOTE "--keep-sequence-bigrams" / "--use-ngrams true" poorer results + runs out of memory
			logger.info("training MALLET LDA with %s" % cmd)
			call(cmd, shell=True)
		else:
			logger.info("skip training step (mallet files already exist: {})".format(repr([self.fcorpusmallet(), self.fstate(), self.fdoctopics(), self.ftopickeys(), self.finferencer()])))

		self.word_topics = self.load_word_topics()

	def load_word_topics(self):
		logger.info("loading assigned topics from %s" % self.fstate())
		wordtopics = numpy.zeros((self.num_topics, self.num_terms), dtype=numpy.float32)
		with utils.smart_open(self.fstate()) as fin:
			_ = next(fin)  # header
			alpha_line = next(fin)
			# logger.info("alpha line: " + repr(alpha_line))
			self.alpha = numpy.array([float(val) for val in alpha_line.split()[2:]])
			assert len(self.alpha) == self.num_topics, "mismatch between MALLET vs. requested topics"
			_ = next(fin)  # beta
			for lineno, line in enumerate(fin):
				line = utils.to_unicode(line)
				doc, source, pos, typeindex, token, topic = line.split(" ")
				if hasattr(self.id2word, 'token2id') and token in self.id2word.token2id:
					tokenid = self.id2word.token2id[token] if hasattr(self.id2word, 'token2id') else int(token)
					wordtopics[int(topic), tokenid] += 1
				else:
					logger.info("TOKEN NOT IN TOKEN2ID!")
					logger.info(repr([lineno, line]))
					logger.info(repr([doc, source, pos, typeindex, token, topic]))
		logger.info("loaded assigned topics for %i tokens" % wordtopics.sum())
		self.wordtopics = wordtopics
		self.print_topics(15)

	# I just want to change the convert_input function so it retains the order in which tokens appeared in the original document (sort by token, not id)
	def convert_input(self, corpus, infer=False):
		"""
		Serialize documents (lists of unicode tokens) to a temporary text file,
		then convert that text file to MALLET format `outfile`.

		"""
		logger.info("serializing temporary corpus to %s" % self.fcorpustxt(infer))
		# write out the corpus in a file format that MALLET understands: one document per line:
		# document id[SPACE]label (not used)[SPACE]whitespace delimited utf8-encoded tokens
		with utils.smart_open(self.fcorpustxt(infer), 'wb') as fout:
			for docno, doc in enumerate(corpus):
				if self.id2word:
					tokens = sum(([self.id2word[tokenid]] * int(cnt) for tokenid, cnt in doc), [])
					if not infer:
						doctext = original_training_docs[corpus.language][docno]['tokenized'] # THIS WILL ONLY WORK IF CORPUS IS THE TRAINING CORPUS
						doctokens = [token for token in doctext.split(' ') if token in tokens]
						tokens = doctokens
				else:
					tokens = sum(([str(tokenid)] * int(cnt) for tokenid, cnt in doc), [])
				fout.write(utils.to_utf8("%s 0 %s\n" % (docno, ' '.join(tokens))))
				# print(utils.to_utf8("%s 0 %s\n" % (docno, ' '.join(tokens))))

		# convert the text file above into MALLET's internal format
		if sys.platform == "win32":
			cmd = self.mallet_path + " import-file --preserve-case --keep-sequence --token-regex (\S+) --input %s --output %s"
		else:
			cmd = self.mallet_path + " import-file --preserve-case --keep-sequence --token-regex '\S+' --input %s --output %s"
		# cmd += " --print-output TRUE"
		if infer:
			cmd += ' --use-pipe-from ' + self.fcorpusmallet()
			cmd = cmd % (self.fcorpustxt(True), self.fcorpusmallet(True) + '.infer')
		else:
			cmd = cmd % (self.fcorpustxt(), self.fcorpusmallet())
		logger.info("converting temporary corpus to MALLET format with %s" % cmd)
		call(cmd, shell=True)

	def __getitem__(self, bow, iterations=100):
		is_corpus, corpus = utils.is_corpus(bow)
		if not is_corpus:
			# query is a single document => make a corpus out of it
			bow = [bow]

		self.convert_input(bow, infer=True)
		cmd = self.mallet_path + " infer-topics --input %s --inferencer %s --output-doc-topics %s --num-iterations %s --doc-topics-threshold %f"
		cmd = cmd % (self.fcorpusmallet(True) + '.infer', self.finferencer(True), self.fdoctopics(True) + '.infer', iterations, 1/(self.num_topics))
		logger.info("inferring topics with MALLET LDA '%s'" % cmd)
		retval = call(cmd, shell=True)
		if retval != 0:
			raise RuntimeError("MALLET failed with error %s on return" % retval)
		result = list(gensim.models.wrappers.ldamallet.read_doctopics(self.fdoctopics(True) + '.infer'))
		return result if is_corpus else result[0]

if __name__ == '__main__':
	# for language in ['es', 'it', 'nl']:
	# for language in ['en','es']:
	# for language in ['en',]:
	# for language in ['all',]:
	for language in ['it',]:
		print("---- Beginning language {} ----".format(language))

		my_training_corpus = build_training_corpus(language)
		with open(os.path.join(training_data_dir, "original_training_fulltext_{}.txt".format(language)), mode='w', encoding='utf8', errors='ignore') as outfile:
			outfile.write("".join([doc['fulltext'] for doc in original_training_docs[language]]))
		print("Wrote copy of (fulltext) training corpus docs to file:", os.path.join(training_data_dir, "original_training_fulltext_{}.txt".format(language)))

		with open(os.path.join(training_data_dir, "original_training_tokenized_{}.txt".format(language)), mode='w', encoding='utf8', errors='ignore') as outfile:
			outfile.write("\n".join([doc['tokenized'] for doc in original_training_docs[language]]))
		print("Wrote copy of (tokenized) training corpus docs to file:", os.path.join(training_data_dir, "original_training_tokenized_{}.txt".format(language)))

		my_test_corpus = build_test_corpus(language, my_training_corpus)
		print("Back in main, done building training and test corpora")
		print("Training:", my_training_corpus)
		print("Test:", my_test_corpus)

		train_id2token = dict([(v,k) for k,v in my_training_corpus.dictionary.token2id.items()])
		test_id2token = dict([(v,k) for k,v in my_test_corpus.dictionary.token2id.items()])

		for token, token_id in my_training_corpus.dictionary.token2id.items():
			if token not in my_test_corpus.dictionary.token2id:
				print("UH OH: token {} exists in training dictionary but not test dictionary!".format(token))
			elif my_test_corpus.dictionary.token2id[token] != token_id:
				print("OH NOES: ids for token {} do not match in training and test dictionaries!".format(token))


		# Trying out the MALLET wrapper corpus...
		mallet_path = os.path.join(os.environ['MALLET_HOME'], "bin", "mallet") # hope this exists

		# print("creating mini training corpus out of first 100 documents")
		# mini_training_corpus = gensim.utils.ClippedCorpus(my_training_corpus, max_docs=5000)
		# mini_training_corpus.language = my_training_corpus.language

		for mallet_topic_count in (100,):
			# train some LDA topics using MALLET
			print("creating the mallet model object with {} topics".format(mallet_topic_count))
			mallet_file_prefix = os.path.join(training_data_dir, "mallet_files", "tworpus_{lang}_{ntopics}_".format(lang=language, ntopics=mallet_topic_count))
			mallet_model = MyLdaMallet(mallet_path,
									   my_training_corpus,
									   # mini_training_corpus,
									   num_topics=mallet_topic_count,
									   id2word=my_training_corpus.dictionary, prefix=mallet_file_prefix)

			# now use the trained model to infer topics on a new document
			# doc = "Don't sell coffee, wheat nor sugar; trade gold, oil and gas instead."
			# # doc = "Non vendere il caffè, grano né zucchero; oro commercio, petrolio e gas, invece"
			# print("query doc:", doc)
			# bow = my_training_corpus.dictionary.doc2bow(my_preprocess(doc))
			# print("query bow:", bow)
			# print(mallet_model[bow])  # print list of (topic id, topic weight) pairs

			# reversed_bow = list(reversed(bow))
			# print("reversed query bow:", reversed_bow)
			# print(mallet_model[reversed_bow])  # print list of (topic id, topic weight) pairs

			print("mallet_model topics:\n", mallet_model.show_topics())

			# # make a small subset of the training corpus to try out the mallet functionality
			# print("creating super mini training corpus out of first 100 documents")
			# super_mini_training_corpus = gensim.utils.ClippedCorpus(my_training_corpus, max_docs=100)
			# super_mini_training_corpus.language = my_training_corpus.language

			num_printed = 0
			for docindex, topic_sim_list in enumerate(mallet_model[my_test_corpus]): # one list of 2-tuples per doc
				if num_printed > 5:
					break
				sims = sorted(topic_sim_list, key=itemgetter(1), reverse=True)
				print("Doc {}:".format(docindex), sims[:3])
				num_printed += 1
