import gensim
import os
import argparse
from pprint import pprint
from author2015 import ArffDict, MyArffEncoder
from config2015 import labels_for_attribute, PERSONALITY_TRAITS


def author_of_tweet(tweet_id, author_dict):
	for author_id in author_dict:
		if tweet_id in author_dict[author_id]:
			return author_id

def doctopics_test(language, input_dir, num_topics):
# def doctopics_test():
# 	language, input_dir, num_topics = get_script_args()
# 	num_topics = int(num_topics)
#
	print("DOCTOPICS: doctopics_test started with args", [language, input_dir, num_topics])

	if "2014" in input_dir:
		print("Found \"2014\" in input_dir name, setting config_year = \"2014\"")
		config_year = '2014'
	else:
		config_year = '2015'

	if language in ('it', 'nl'): # these two languages are not labeled with author's age group
		class_attributes = ('gender', 'extroverted', 'stable', 'agreeable', 'conscientious', 'open')
	else: # 'es' and 'en' are labeled with age group
		if config_year == '2015':
			class_attributes = ('gender', 'age_group', 'extroverted', 'stable', 'agreeable', 'conscientious', 'open')
		else:
			class_attributes = ('gender', 'age_group')


	doctopics_filename = os.path.join(input_dir, "{lang}_{num_topics}_doctopics.txt".format(lang=language, num_topics=num_topics))
	doctopics_results = list(gensim.models.wrappers.ldamallet.read_doctopics(doctopics_filename, eps=1/(2*num_topics)))
	print(len(doctopics_results), "rows read from doctopics file", doctopics_filename)

	author_offsets = {}
	doctopics_keys_filename = os.path.join(input_dir, "all_{lang}_tweets_keys.txt".format(lang=language))
	with open(doctopics_keys_filename, mode='r', encoding='utf8') as author_offsets_file:
		for line in author_offsets_file.readlines():
			author_id, first, last = line.strip().split(',')[:3]
			author_offsets.update({author_id: range(int(first), int(last))})

	# write an arff file with the attributes [author id] [mallet topic features] [class label]
	arff_data = []
	for tweet_index, topic_list in enumerate(doctopics_results):
		author_id = author_of_tweet(tweet_index, author_offsets)
		topic_dict = dict(topic_list)
		arff_data.append([author_id] + [topic_dict.get(topic_index, 0) for topic_index in range(0, num_topics)] + ['?'])

	for class_attribute in class_attributes:
		if class_attribute in PERSONALITY_TRAITS:
			class_attribute_spec = (class_attribute, 'NUMERIC')
		else:
			class_attribute_spec = (class_attribute, sorted(labels_for_attribute[class_attribute]))

		arff_filepath = os.path.join(input_dir, "{lang}_{attr}_test_mallet.arff".format(lang=language, attr=class_attribute)) #doctopics_filename.replace("all_", "test_").replace("doctopics", class_attribute).replace(".txt", ".arff")

		training_arff = ArffDict(
			relation = os.path.splitext(os.path.basename(arff_filepath))[0], # something like "nl_gender_test_mallet"
			attributes = [('author_id', [author for author in author_offsets.keys()])] +
						 [("mallet{0}".format(str(topic).zfill(3)), 'NUMERIC') for topic in range(0, num_topics)] +
						 [class_attribute_spec],
			data = arff_data  )

		arff_enc = MyArffEncoder()

		print("\nDOCTOPICS: MyArffEncoder writing instances (including author_id and class label attributes) to ARFF file...")

		with open(arff_filepath, mode='w', encoding='utf8') as arff_file:
			for instance in arff_enc.iter_encode(training_arff):
				arff_file.write(instance + u'\n')
		print("DOCTOPICS: MyArffEncoder wrote training data file:", arff_filepath)

	# write an arff file with ONLY the [mallet topic features] as attributes
	arff_data = []
	for tweet_index, topic_list in enumerate(doctopics_results):
		topic_dict = dict(topic_list)
		arff_data.append([topic_dict.get(topic_index, 0) for topic_index in range(0, num_topics)])

	arff_filepath = os.path.join(input_dir, "{lang}_test_justmallet.arff".format(lang=language))

	training_arff = ArffDict(
		relation = os.path.splitext(os.path.basename(arff_filepath))[0], # something like "nl_gender_test_mallet"
		attributes = [("mallet{0}".format(str(topic).zfill(3)), 'NUMERIC') for topic in range(0, num_topics)],
		data = arff_data  )

	arff_enc = MyArffEncoder()
	print("\nDOCTOPICS: MyArffEncoder writing instances (mallet topic attributes only) to ARFF file...")

	with open(arff_filepath, mode='w', encoding='utf8') as arff_file:
		for instance in arff_enc.iter_encode(training_arff):
			arff_file.write(instance + u'\n')
	print("DOCTOPICS: MyArffEncoder wrote training data file:", arff_filepath)
