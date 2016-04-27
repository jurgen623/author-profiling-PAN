import time
import sys
import argparse
import tempfile
# from operator import itemgetter
# from nltk import FreqDist, ngrams
# import arff # the liac-arff python library
from pprint import pformat

import psutil
from meminfo import pprint_ntuple

from custom_features import *
from process_tweets import *
import weka_classify # separate file to hold imports and config settings for weka/java

# tweetwise_split = 'train' # train: use first 2/3 of every author's tweets. test: use last 1/3 of every author's tweets. any other value: use all tweets.


def get_script_args():
	argparser = argparse.ArgumentParser(description = "Train a classifier on specified input dataset and save a model to an output directory.")
	argparser.add_argument('inputdir', help="path to directory containing unpacked input dataset", default='training_input')
	argparser.add_argument('outputdir', help="path to an empty directory where the model will be saved", default='training_output')
	argparser.add_argument('modeldir', help="path to a directory where models will be saved for the testing phase", default='shared_data')
	argparser.add_argument('cvfolds', help="command-line setting to pass to weka classifiers for cross-validation. should be -1 for no cross-validation, or positive number for number of folds.", default=-1)
	argparser.add_argument('targetattrs', help="period-separated list of attribute names to target on this run or 'all' to test all", default='all')
	argparser.add_argument('featureprefixes', help="period-separated list of prefixes to feature names that should be used, ie 'topic.high.low' or 'all'")
	argparser.add_argument('combinetweets', help="y or n: if y, combine all of an author's tweets into single instances", default="y")
	argparser.add_argument('numericclassifier', help="string name for desired numeric classifier settings to load from weka_classify.py (use 'default' to use the value of the selected_numeric_classifier variable)", default="default")
	argparser.add_argument('nominalclassifier', help="string name for desired nominal classifier settings to load from weka_classify.py (use 'default' to use the value of the selected_nominal_classifier variable)", default="default")
	argparser.add_argument('split', help="train: use first 2/3 of every author's tweets. test: use last 1/3 of every author's tweets. any other value: use all tweets", default='all')
	args = argparser.parse_args()
	return args.inputdir, args.outputdir, args.modeldir, args.cvfolds, args.targetattrs, args.featureprefixes, args.combinetweets, args.numericclassifier, args.nominalclassifier, args.split

	
def arff_encoder_instances(unlabeled_instance_dict, authors):
	# keys of unlabeled_instance_dict are author_ids, values are lists of numpy ndarrays (where each ndarray is a 2000+ element featurized tweet)
	# keys of authors are author_ids, values are LabeledAuthor objects
	instances = []
	for author_id in sorted(authors.keys()):
		#author_label = authors[author_id].get_label(attribute)
		instances.extend([[author_id] + list(instance_nolabel) + ['?'] for instance_nolabel in unlabeled_instance_dict[author_id]])
	print(len(instances[0]), "elements in first row of labeled_instances")
	return instances


def __main__():
	print("Classifier training program running in directory", os.path.abspath(os.curdir))
	print("***", time.ctime())

	# Uncomment to get input and output directory names from command line arguments to this script
	inputdir, outputdir, modeldir, cvfolds, targetattrs, featureprefixes, combinetweets, numericclassifier, nominalclassifier, tweetwise_split = get_script_args()

	inputdir = os.path.join(os.curdir, inputdir)
	outputdir = os.path.join(os.curdir, outputdir)
	modeldir = os.path.join(os.curdir, modeldir)

	if os.path.exists(inputdir) and os.path.isdir(inputdir):
		print("Found training inputdir", inputdir)
		if "2014" in inputdir:
			print("Found \"2014\" in inputdir name, setting config_year = \"2014\"")
			config_year = '2014'
		elif "2016" in inputdir:
			print("Found \"2016\" in inputdir name, setting config_year = \"2014\" = \"2016\"")
			config_year = '2014'
		else:
			config_year = '2015'
	else:
		print("Couldn't find training inputdir", inputdir)
		return False

	if os.path.exists(outputdir) and os.path.isdir(outputdir):
		print("Found training outputdir", outputdir)
	else:
		print("Couldn't find training outputdir", outputdir)
		return False

	if os.path.exists(modeldir) and os.path.isdir(modeldir):
		print("Found train/test shared modeldir", modeldir)
	else:
		print("Couldn't find train/test shared modeldir", modeldir)
		return False

	tempdir = tempfile.mkdtemp(suffix='', prefix="temptrain_", dir=os.curdir)
	print("Using directory", tempdir, "for temporary working files")

	if combinetweets.lower() == "n" or combinetweets.lower() == "no":
		print("\nReceived option '-combinetweets n(o)': Do not combine multiple tweets into single instances")
		combinetweets = False
	else:
		print("\nCombine multiple tweets into single instances (default)")
		combinetweets = True

	numeric_classifier_config = weka_classify.numeric_classifier_configs[weka_classify.selected_numeric_classifier]
	if numericclassifier.lower() == "default":
		print("\nUsing default numeric classifier configuration specified in selected_numeric_classifier of weka_classify.py:", weka_classify.selected_numeric_classifier)
		print(weka_classify.numeric_classifier_configs[weka_classify.selected_numeric_classifier])
	elif numericclassifier not in weka_classify.numeric_classifier_configs.keys():
		print("\nUnable to find requested numeric classifier '{requested}' in weka_classify.numeric_classifier_configs".format(requested=numericclassifier))
		print(" Available numeric classifier keys:", weka_classify.numeric_classifier_configs.keys())
		print("Using numeric classifier configuration specified in selected_numeric_classifier of weka_classify.py:", weka_classify.selected_numeric_classifier)
		print(weka_classify.numeric_classifier_configs[weka_classify.selected_numeric_classifier])
	else:
		print("\nUsing requested numeric classifier '{requested}'".format(requested=numericclassifier))
		print(weka_classify.numeric_classifier_configs[numericclassifier])
		numeric_classifier_config = weka_classify.numeric_classifier_configs[numericclassifier]

	nominal_classifier_config = weka_classify.nominal_classifier_configs[weka_classify.selected_nominal_classifier]
	if nominalclassifier.lower() == "default":
		print("\nUsing default nominal classifier configuration specified in selected_nominal_classifier of weka_classify.py:", weka_classify.selected_nominal_classifier)
		print(weka_classify.nominal_classifier_configs[weka_classify.selected_nominal_classifier])
	elif nominalclassifier not in weka_classify.nominal_classifier_configs.keys():
		print("\nUnable to find requested nominal classifier '{requested}' in weka_classify.nominal_classifier_configs".format(requested=nominalclassifier))
		print(" Available nominal classifier keys:", weka_classify.nominal_classifier_configs.keys())
		print("Using nominal classifier configuration specified in selected_nominal_classifier of weka_classify.py:", weka_classify.selected_nominal_classifier)
		print(weka_classify.nominal_classifier_configs[weka_classify.selected_nominal_classifier])
	else:
		print("\nUsing requested nominal classifier '{requested}'".format(requested=nominalclassifier))
		print(weka_classify.nominal_classifier_configs[nominalclassifier])
		nominal_classifier_config = weka_classify.nominal_classifier_configs[nominalclassifier]

	if tweetwise_split == 'test':
		print('\nUsing "test" portion of each author\'s tweets (last one-third of tweets, as ordered in input xml file)')
	elif tweetwise_split == 'train':
		print('\nUsing "train" portion of each author\'s tweets (first two-thirds of tweets, as ordered in input xml file)')
	elif tweetwise_split == '100':
		print('\nUsing (up to) first 100 tweets, as ordered in input xml file')
	else:
		print('\nUsing all available tweets per author, not "train" (first two-thirds) or "test" (last one-third) portions')

	selected_feature_prefixes = all_feature_prefixes
	if featureprefixes.lower() == "all":
		print("\nReceived option '-featureprefixes all': Using all available feature types:", selected_feature_prefixes)
	elif featureprefixes in all_feature_prefixes:
		print("\nReceived option '-featureprefixes", featureprefixes, "': Using only features beginning with:", featureprefixes)
		selected_feature_prefixes = [featureprefixes]
	elif "." in featureprefixes:
		print("\nReceived option '-featureprefixes {0}'".format(featureprefixes))
		featureprefixes = featureprefixes.split(".")
		print("Requested to use features beginning with one of:", featureprefixes)
		featureprefixes = [prefix for prefix in all_feature_prefixes if prefix in featureprefixes]
		if len(featureprefixes) < 1:
			print("Requested feature prefixes not in available list; using all features instead")
		else:
			selected_feature_prefixes = featureprefixes
			print("Using features beginning with one of:", selected_feature_prefixes)

	cvfolds = int(cvfolds)
	if cvfolds != -1:
		cv_argument_str = "-x {0}".format(cvfolds)
		print("\nUsing {0}-fold cross-validation for weka classifier commands".format(cvfolds))
	else:
		cv_argument_str = "-no-cv"
		print("\nUsing no cross-validation for weka classifier commands")

	print("\nMaking predictions for attributes:", targetattrs)
	targetattrs = targetattrs.split(".")
	if 'all' in targetattrs:
		targetattrs = ['gender', 'age_group', 'extroverted', 'stable', 'agreeable', 'conscientious', 'open']

	external_commands_list = []
	if sys.platform == "win32":
		external_commands_filename = os.path.join(os.curdir, "shell_actions_2016.bat")
	else:
		external_commands_filename = os.path.join(os.curdir, "shell_actions_2016.sh")

	input_xml_files = get_input_files(inputdir, '.xml')
	xml_filenames_by_language, userids_by_language = group_by_language(input_xml_files)
	# xml_filenames_by_language: keys are ('en', 'es', 'it', 'nl'), values are lists of .xml filenames
	# userids_by_language: keys are ('en', 'es', 'it', 'nl'), values are lists of userid strings

	for language in xml_filenames_by_language:
		# Write intermediate .txt files holding cleaned-up and tokenized tweet content, one tweet per line of a file
		write_preprocessed_tweets(language, xml_filenames_by_language[language], tempdir, tweetwise_split)

	input_truth_files = get_input_files(inputdir, "truth.txt")
	labeled_authors = get_labeled_authors(input_truth_files, userids_by_language, tempdir, force_lowercase=False, config_year=config_year)
	# labeled_authors:
	# keys are ('en', 'es', 'it', 'nl'),
	# values are dictionaries whose keys are author userid strings, values are LabeledAuthor objects

	for training_language in sorted(labeled_authors.keys()):
		training_authors = labeled_authors[training_language]
		if len(training_authors) < 1:
			continue
		print("*** Starting {0} at {1}".format(training_language, time.ctime()))


		# generate a text file that mallet can import:
		# one "document" (tweet) per line. format is:
		# 	document-id[SPACE]label[SPACE]whitespace-delimited-utf8-encoded-tokens
		# document-id will be the index of that tweet in a sorted list of all tweets in this language,
		# label will be the author's userid (I guess?)
		# the last field is the tweet content, which we have already preprocessed

		# ONLY FOR USING MALLET FEATURES:
		"""
		alltweets = []
		num_topics = 100

		author_tweet_ranges = dict.fromkeys(training_authors.keys()) # dictionary to keep track of the first and last tweet indexes belonging to each author
		tweets_written = 0
		for author_id in sorted(training_authors.keys()):
			if len(training_authors[author_id].tweets) > 0:
				author_tweet_ranges[author_id] = {'first': tweets_written, 'last': tweets_written + len(training_authors[author_id].tweets)}
			for tweet in training_authors[author_id].tweets:
				alltweets.append([tweets_written, author_id, tweet])
				tweets_written += 1
		alltweets_filename = os.path.join(modeldir, "all_{lang}_tweets.txt".format(lang=training_language))
		alltweets_keys_filename = os.path.join(modeldir, "all_{lang}_tweets_keys.txt".format(lang=training_language))
		with open(alltweets_filename, mode='w', encoding='utf8') as alltweets_file:
			alltweets_file.write("\n".join(["{}\t{}\t{}".format(num, label, text) for num, label, text in alltweets]))
			print("Wrote", len(alltweets), "tweets to", alltweets_filename, "for processing in MALLET")

		with open(alltweets_keys_filename, mode='w', encoding='utf8') as alltweets_keys_file:
			for author_id, range_data in sorted(author_tweet_ranges.items()):
				author_data = training_authors[author_id]

				if config_year == '2015':
					truth_data = "{gender},{age},{},{},{},{},{}".format(
						gender=author_data.gender, age=author_data.age_group, *author_data.traits)
					alltweets_keys_file.write( "{},{},{},{}\n".format(author_id, range_data['first'], range_data['last'], truth_data) )
				else:
					truth_data = "{gender},{age}".format(gender=author_data.gender, age=author_data.age_group)
					alltweets_keys_file.write( "{},{},{},{}\n".format(author_id, range_data['first'], range_data['last'], truth_data) )

			print("Wrote first and last tweet indexes for all {lang} authors to file".format(lang=training_language), alltweets_keys_filename)
		"""

		if sys.platform == "win32":
			external_commands_list.append('make_mallet_files.bat {lang} {mallet_dest} {arff_dest} {num_topics}'.format(lang=training_language, mallet_dest=modeldir, arff_dest=outputdir, num_topics=num_topics))
			# instead of putting this in shell_actions.bat, call using the pycharm run configuration, so we can call doctopics_util.py as its own configuration, using debugger and stuff
			# print('FAKE COMMAND: make_mallet_files.bat {lang} {mallet_dest} {arff_dest} {num_topics}'.format(lang=training_language, mallet_dest=modeldir, arff_dest=outputdir, num_topics=num_topics))
		else:
			external_commands_list.append('sh make_mallet_files.sh {lang} {mallet_dest} {arff_dest} {num_topics}'.format(lang=training_language, mallet_dest=modeldir, arff_dest=outputdir, num_topics=num_topics))

		if config_year == '2015' and training_language in ('it', 'nl'): # these two languages are not labeled with author's age group
			training_attributes = ('gender', 'extroverted', 'stable', 'agreeable', 'conscientious', 'open')
		else: # 'es' and 'en' are labeled with age group
			if config_year == '2015':
				training_attributes = ('gender', 'age_group', 'extroverted', 'stable', 'agreeable', 'conscientious', 'open')
			else:
				training_attributes = ('gender', 'age_group')

		# From attributes that are possible to predict for this language/configuration, take only those specified as a script argument
		training_attributes = [attribute for attribute in training_attributes if attribute in targetattrs]

		feature_names = []
		if 'topic' in selected_feature_prefixes:
			feature_names += ["topic{0}".format(str(topic).zfill(4)) for topic in range(0,max_topic_index+1)]
		for name in custom_feature_names:
			for prefix in selected_feature_prefixes:
				if name.startswith(prefix):
					feature_names += [name]
					continue
		print("TRAIN: Found", len(feature_names), "feature names matching selected feature prefixes:", selected_feature_prefixes)
		print(", ".join([name for name in feature_names if not name.startswith("topic")]))
		if 'topic' in selected_feature_prefixes:
			print(len([name for name in feature_names if name.startswith("topic")]), "topics features\n")


		training_instances_nolabels = extract_instance_features(training_authors, feature_names, combinetweets)

		weka_features = get_weka_features(training_authors, feature_names)

		training_arff = ArffDict(
			relation = training_language,
			attributes = weka_features + [('CLASS', 'NUMERIC')], # ('CLASS', 'NUMERIC') won't be written to any file (it's just a placeholder)
			data = arff_encoder_instances(training_instances_nolabels, training_authors),
			)

		for class_attribute in training_attributes:
			print("*** Starting {0}, {1} at {2}".format(training_language, class_attribute, time.ctime()))
			training_labels = labels_for_attribute[class_attribute]
			language_class_prefix = "{lang}_{classattr}".format(lang=training_language, classattr=class_attribute)

			train_filename = language_class_prefix + '_train_py.arff'
			train_filepath = os.path.join(outputdir, train_filename)

			# filtered_train_filepath = os.path.join(modeldir, train_filename)

			print("TRAIN: Using", len(weka_features), "features, not including class attribute", class_attribute)

			arff_enc = MyArffEncoder()

			if class_attribute in PERSONALITY_TRAITS:
				training_labels = 'NUMERIC'
			else:
				training_labels = sorted(training_labels)
			training_arff['relation'] = language_class_prefix
			training_arff['description'] = "PAN Author Profiling Task 2016: " + language_class_prefix
			training_arff['attributes'][-1] = (class_attribute, training_labels)

			print("TRAIN: MyArffEncoder labeling instances with authors' values of class attribute", class_attribute)
			for data_row in training_arff['data']:
				# the author_id that wrote this tweet is the first element in the row of data: data[0]
				author_label = training_authors[data_row[0]].get_label(class_attribute)
				data_row[-1] = author_label

			print("TRAIN: MyArffEncoder writing instances to ARFF file...")
			with open(train_filepath, mode='w', encoding='utf8') as train_file:
				for instance in arff_enc.iter_encode(training_arff):
					train_file.write(instance + u'\n')
			print("TRAIN: MyArffEncoder wrote training data file:", train_filepath)

			short_classifier_name = ""

			# for data_source in ("py", "mallet"):
			for data_source in ("py",):
				train_filename = "{prefix}_train_{source}.arff".format(prefix=language_class_prefix, source=data_source)
				train_filepath = os.path.join(outputdir, train_filename)
				filtered_train_filepath = os.path.join(modeldir, train_filename)

				if class_attribute in PERSONALITY_TRAITS:
					# Get the variable part of the multifilter command from our chosen config in weka_classify.py
					multifilter_cmd = numeric_classifier_config['multifilter_cmd']
					if multifilter_cmd != '':
						multifilter_cmd += " -i {input_arff_file} -o {filtered_arff_file} -decimal 6".format(input_arff_file=train_filepath, filtered_arff_file=filtered_train_filepath)

					# Get the variable part of the classifier training command from our chosen config in weka_classify.py
					train_cmd_format = numeric_classifier_config["train_cmd_format"]
					short_classifier_name = numeric_classifier_config["name"]
				else:
					filter_cmd_format = "weka.filters.unsupervised.attribute.AddValues -C last -S -L {labels}"
					filter_cmd = filter_cmd_format.format(labels=",".join(training_labels))

					# Get the variable part of the multifilter command from our chosen config in weka_classify.py
					multifilter_format = nominal_classifier_config['multifilter_cmd']
					if multifilter_format != '':
						multifilter_cmd = multifilter_format.format(other_filter_cmd=filter_cmd)
						multifilter_cmd += " -i {input_arff_file} -o {filtered_arff_file} -decimal 6".format(input_arff_file=train_filepath, filtered_arff_file=filtered_train_filepath)

					# Get the variable part of the classifier training command from our chosen config in weka_classify.py
					train_cmd_format = nominal_classifier_config["train_cmd_format"]
					short_classifier_name = nominal_classifier_config["name"]

				if multifilter_cmd != '':
					print("\nTRAIN: Adding command to {file}: {cmd}".format(file=external_commands_filename, cmd=multifilter_cmd))
					external_commands_list.append(multifilter_cmd)
				else:
					# if we're not doing an extra filter step before sending the arff file into the classifier, just move the
					# training arff file we wrote with MyArffEncoder into the shared model directory
					if sys.platform == "win32":
						external_commands_list.append("MOVE {source} {dest}".format(source=train_filepath, dest=filtered_train_filepath))
						# print("FAKE COMMAND: MOVE {source} {dest}".format(source=train_filepath, dest=filtered_train_filepath))
					else:
						external_commands_list.append("mv {source} {dest}".format(source=train_filepath, dest=filtered_train_filepath))

				model_filename = "weka_{lang_class}_{classifier}_{source}.model".format(lang_class=language_class_prefix, classifier=short_classifier_name, source=data_source)
				model_filepath = os.path.join(modeldir, model_filename)

				train_cmd = train_cmd_format.format(output_model_file=model_filepath,
													filtered_arff_file=filtered_train_filepath,
													crossval=cv_argument_str)
				print("\nTRAIN: Adding command to {file}: {cmd}".format(file=external_commands_filename, cmd=train_cmd))
				external_commands_list.append(train_cmd)

		print('\nMEMORY\n------')
		pprint_ntuple(psutil.virtual_memory())

	with open(external_commands_filename, mode='w', encoding='utf8') as external_commands_file:
		if sys.platform == "win32":
			for cmd in external_commands_list:
				if cmd.startswith("weka"):
					cmd = 'java -cp "{wekahome}" {command} '.format(wekahome=os.environ['WEKAHOME'], command=cmd)
				external_commands_file.write(cmd + '\n')
				external_commands_file.write("ECHO %time% \n")
		else:
			external_commands_file.write("#!/bin/bash \n")
			for cmd in external_commands_list:
				external_commands_file.write('echo "{command}" \n'.format(command=cmd))
				if cmd.startswith("weka"):
					cmd = 'java -Xmx3000M -cp {wekahome} {command} '.format(wekahome=os.environ['WEKAHOME'], command=cmd)
				external_commands_file.write(cmd + '\n')
				external_commands_file.write('date  \n')
			os.chmod(external_commands_filename, 0o755)
	print("\nTRAIN: Wrote external training commands to file:", external_commands_filename)

	print("***", time.ctime())
	print('TRAIN: Removing temporary directory', tempdir)
	for f in os.listdir(tempdir):
		os.remove(os.path.join(tempdir, f))
	os.rmdir(tempdir)

	# --- Only needed to run this once, to get a set of files to compare to with accuracy.py
	# author_truth_dict = {}
	# for training_language in sorted(labeled_authors.keys()):
	# 	training_authors = labeled_authors[training_language]
	# 	for author_id, author_data in training_authors.items():
	# 		author_truth_dict[author_id] = author_data.get_truth_dict()
	# with open(os.path.join(os.curdir, "author_truths.txt"), mode='w', encoding='utf8') as author_truths_file:
	# 	author_truths_file.write(pformat(author_truth_dict))
	# print("Wrote", os.path.join(os.curdir, "author_truths.txt"))


__main__()
