import os
import gensim
import logging
from pprint import pprint as pprint
from author_truths import author_truth_dict
from arktwokenizepy import twokenize
from urllib.parse import urlparse, urlunparse
#from build_corpus import training_data_dir, test_data_dir, get_sim, MyTrainingCorpus, MyTestCorpus, MyLdaMallet
from build_corpus import *

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Just import these from build_corpus
# training_data_dir = os.path.join(os.curdir, "gensim", "en_moretweets", "sentiment140")
# test_data_dir = os.path.join(os.curdir, "gensim", "pan15txt")


def load_word2vec_model(path):
	model = gensim.models.Word2Vec.load_word2vec_format(path, binary=True) # C binary format
	return model

def dictionary_id2token(dictionary_obj, word_id):
	for k, v in iter(dictionary_obj.token2id.items()):
		if v == word_id:
			return k
	return None

def lda_stuff(language, training_corpus, test_corpus):
	if os.path.exists(os.path.join(training_data_dir, "train_tfidf_model.svmlight")):
		print("Loading TFIDF training corpus from file", os.path.join(training_data_dir, "train_tfidf_model.svmlight"))
		tfidf_model = gensim.models.TfidfModel.load(os.path.join(training_data_dir, "train_tfidf_model.svmlight"), mmap='r')
	else:
		print("Existing TFIDF training corpus not found, creating now...")
		tfidf_model = gensim.models.TfidfModel(training_corpus) # step 1 -- initialize a model
		tfidf_model.save(os.path.join(training_data_dir, "train_tfidf_model.svmlight"), separately=None)
	# 2015-05-06 13:21:00,420 : INFO : saving TfidfModel object under .\gensim\en_moretweets\sentiment140\train_tfidf_model.svmlight, separately None


	training_corpus_tfidf = tfidf_model[training_corpus] # map all the documents from the training set into the TF-IDF model we just built
	test_corpus_tfidf = tfidf_model[test_corpus] # map documents from test set into the TF-IDF model built from the training set

	# Just for debugging time:
	docs_count = 0
	nonempty_docs_count = 0
	for test_doc in iter(test_corpus):
		if nonempty_docs_count > 5 or docs_count > 100:
			break

		if len(test_doc) > 0:
			print("\nTest corpus document", docs_count, test_doc)
			test_tokens = [(word_id, train_id2token[word_id], int(word_count)) for word_id, word_count in test_doc if word_id in train_id2token]
			unmatched_test_tokens = [(word_id, word_count) for word_id, word_count in test_doc if word_id not in train_id2token]
			print("Matched test tokens:", test_tokens)
			if len(unmatched_test_tokens) > 0:
				print("VTF: unmatched tokens in test doc {}:".format(docs_count), unmatched_test_tokens)
			if len(test_tokens) > 0:
				nonempty_docs_count += 1
		docs_count += 1

	# Just for debugging time:
	docs_count = 0
	for test_doc in test_corpus_tfidf:
		if docs_count > 10:
			break
		print(test_doc)
		docs_count += 1

	print("\nReady to test similarity query index!")

	silly_query = "Its mission is to help NLP practitioners try out popular topic modelling algorithms on large datasets easily"

	if os.path.exists(os.path.join(training_data_dir, "lda_model_20topics.svmlight")):
		print("Loading LDA model from file", os.path.join(training_data_dir, "lda_model_20topics.svmlight"))
		lda_model = gensim.models.LdaModel.load(os.path.join(training_data_dir, "train_tfidf_model.svmlight"), mmap='r')
	else:
		print("Existing LDA model not found, creating now...")
		lda_model = gensim.models.LdaModel(list(training_corpus_tfidf), id2word=training_corpus.dictionary, num_topics=20)
		print("Saving LDA model to file")
		lda_model.save(os.path.join(training_data_dir, "lda_model_20topics.svmlight"), separately=None)
	# 2015-05-06 13:12:33,523 : INFO : saving LdaState object under .\gensim\en_moretweets\sentiment140\lda_model_20topics.svmlight.state, separately None
	# 2015-05-06 13:12:33,551 : INFO : saving LdaModel object under .\gensim\en_moretweets\sentiment140\lda_model_20topics.svmlight, separately None
	# 2015-05-06 13:12:33,551 : INFO : not storing attribute state
	# 2015-05-06 13:12:33,551 : INFO : not storing attribute dispatcher

	print("Applying LDA model to training corpus")
	training_corpus_lda = lda_model[training_corpus_tfidf]

	print("Applying LDA model to test corpus")
	test_corpus_lda = lda_model[test_corpus_tfidf]

	if os.path.exists(os.path.join(training_data_dir, "lda_20topics_index")):
		print("Attempting to load LDA similarity index from files starting with:\n", os.path.join(training_data_dir, "lda_20topics_index"))
		lda_index = gensim.similarities.Similarity.load(os.path.join(training_data_dir, "lda_20topics_index.0"), mmap='r')
	else:
		# lda_index = gensim.similarities.Similarity(os.path.join(training_data_dir, "lda_20topics_index"), training_corpus_lda, num_features=lda_model.num_topics, num_best=20)
		lda_index = gensim.similarities.MatrixSimilarity(training_corpus_lda, num_features=lda_model.num_topics, num_best=20)
		lda_index.save(os.path.join(training_data_dir, "lda_20topics_index"), separately=None)

	test_query_docs = []
	test_docs_processed = 0
	while len(test_query_docs) < 10:
		test_query_docs.append( next(iter(test_corpus_lda)) )
		test_docs_processed += 1
	test_similarities = [sim for sim in lda_index[test_query_docs]]
	pprint(test_similarities)
	print(test_docs_processed, "total test query docs processed")

	# compute similarity of test corpus documents to training corpus documents in the lda model space
	for test_doc_result in lda_index[test_query_docs]: # for each of the test documents we compared with the lda-modeled training docs
		print( ", ".join(
			["{docid}: {sim:0.8}".format(docid=training_doc_id, sim=training_doc_similarity)
			 for training_doc_id, training_doc_similarity in test_doc_result]
		))
	print(test_docs_processed, "total test query docs processed")
	pprint(lda_model.inference(test_query_docs))
	# compute the distribution over lda model topics for each of the test corpus documents
	for test_doc_result in lda_model.inference(test_query_docs)[0]: # the [0] is because we want to throw out something else returned by the function
		print( ", ".join(
			["{topicid}: {score:0.8}".format(topicid=topic_id, score=topic_relevance)
			 for topic_id, topic_relevance in enumerate(test_doc_result)]
		))
	print(test_docs_processed, "total test query docs processed")

def random_projections(language, training_corpus, test_corpus):
	print("\nCreating random projections model for language", language)
	rpmodel = gensim.models.RpModel(training_corpus, id2word=training_corpus.dictionary, num_topics=300)
	rpmatrix = rpmodel[[doc for doc in test_corpus]]
	print(list(rpmatrix)[:10])

if __name__ == '__main__':
	# for language in ['en', 'es', 'it', 'nl']:
	for language in ['en', 'es']:
	# for language in ['en',]:
		print("\n---- Beginning language {} ----".format(language))

		train_dictionary_filename = os.path.join(training_data_dir, "train_gensim_dictionary_{}.dict".format(language))
		train_corpus_filename = os.path.join(training_data_dir, "train_gensim_corpus_{}.svmlight".format(language))
		
		test_dictionary_filename = os.path.join(test_data_dir, "test_gensim_dictionary_{}.dict".format(language))
		test_corpus_filename = os.path.join(test_data_dir, "test_gensim_corpus_{}.svmlight".format(language))
		
		
		my_training_corpus = gensim.corpora.SvmLightCorpus(train_corpus_filename)
		train_dictionary = gensim.corpora.Dictionary.load(train_dictionary_filename, mmap=None)
		my_training_corpus.dictionary = train_dictionary
		print("Training dictionary:", train_dictionary)

		my_test_corpus = gensim.corpora.SvmLightCorpus(test_corpus_filename)
		my_test_corpus.dictionary = train_dictionary
		# test_dictionary = gensim.corpora.Dictionary.load(test_dictionary_filename, mmap=None)
		# print("Test dictionary:", test_dictionary)

		train_id2token = dict([(v,k) for k,v in my_training_corpus.dictionary.token2id.items()])
		# test_id2token = dict([(v,k) for k,v in test_dictionary.token2id.items()])

		for token, token_id in my_training_corpus.dictionary.token2id.items():
			if token not in my_test_corpus.dictionary.token2id:
				print("UH OH: token {} exists in training dictionary but not test dictionary!".format(token))
			elif my_test_corpus.dictionary.token2id[token] != token_id:
				print("OH NOES: ids for token {} do not match in training and test dictionaries!".format(token))


		# GENSIM LDA STUFF HAPPENS HERE ....
		# lda_stuff(language, my_training_corpus, my_test_corpus)
		# GENSIM LDA STUFF NOW DONE

		# GENSIM RANDOM PROJECTIONS STUFF HAPPENS HERE ....
		# random_projections(language, my_training_corpus, my_test_corpus)
		# GENSIM RANDOM PROJECTIONS STUFF NOW DONE

		# Trying out the MALLET wrapper corpus...
		mallet_path = os.path.join(os.environ['MALLET_HOME'], "bin", "mallet") # hope this exists

		mallet_model_prefix = os.path.join(training_data_dir, "mallet_files", "mallet_" + language + "_")

		# try instantiating the mallet model object from files that already exist on disk
		print("creating the mallet model object from existing files located at {}*".format(mallet_model_prefix))
		mallet_model = MyLdaMallet(mallet_path,
								   None, # if corpus is None, LdaMallet won't call train(corpus) in its __init__
								   #my_training_corpus,
								   num_topics=200,
								   id2word=my_training_corpus.dictionary, prefix=mallet_model_prefix)
		mallet_model.word_topics = mallet_model.load_word_topics() # have to do this manually if .train() isn't going to be called

		# quick test if the model is working...

		# now use the trained model to infer topics on a new document
		doc = "Don't sell coffee, wheat nor sugar; trade gold, oil and gas instead."
		# doc = "Non vendere il caffè, grano né zucchero; oro commercio, petrolio e gas, invece"
		print("query doc:", doc)
		bow = my_training_corpus.dictionary.doc2bow(my_preprocess(doc))
		print("query bow:", bow)
		print(mallet_model[bow])  # print list of (topic id, topic weight) pairs

		# reversed_bow = list(reversed(bow))
		# print("reversed query bow:", reversed_bow)
		# print(mallet_model[reversed_bow])  # print list of (topic id, topic weight) pairs

		print("mallet_model topics:\n", mallet_model.show_topics())


		for docindex, topic_sim_list in enumerate(mallet_model[my_test_corpus]): # one list of 2-tuples per doc in mini_training_corpus
			sims = sorted(topic_sim_list, key=itemgetter(1), reverse=True)
			print("Doc {}:".format(docindex), sims[:3])