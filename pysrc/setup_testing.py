import argparse
import tempfile
import time
import sys
from subprocess import call
# from nltk import FreqDist, ngrams
from operator import itemgetter
from numpy import mean, median, std, array
# import arff # the liac-arff python library

from custom_features import *
from process_tweets import *
from author_truths import author_truth_dict
import weka_classify # separate file to hold imports and config settings for weka/java
from doctopics_test import doctopics_test

export_combined_arff = False
test_combined_arff = True
known_labels = False
# tweetwise_split = 'test' # train: use first 2/3 of every author's tweets. test: use last 1/3 of every author's tweets. any other value: use all tweets.

def get_script_args():
	argparser = argparse.ArgumentParser(description = "Test a classifier on specified input dataset and save predictions in XML files in an output directory.")
	argparser.add_argument('inputdir', help="path to directory containing unpacked input dataset", default='testing_input')
	argparser.add_argument('outputdir', help="path to an empty directory where the model will be saved", default='testing_output')
	argparser.add_argument('modeldir', help="path to a directory where models were saved from the training phase", default='shared_data')
	argparser.add_argument('targetattrs', help="period-separated list of attribute names to target on this run or 'all' to test all", default='all')
	argparser.add_argument('featureprefixes', help="period-separated list of prefixes to feature names that should be used, ie 'topics.high.low' or 'all'")
	argparser.add_argument('combinetweets', help="y or n: if y, combine all of an author's tweets into single instances", default="y")
	argparser.add_argument('numericclassifier', help="string name for desired numeric classifier settings to load from weka_classify.py (use 'default' to use the value of the selected_numeric_classifier variable)", default="default")
	argparser.add_argument('nominalclassifier', help="string name for desired nominal classifier settings to load from weka_classify.py (use 'default' to use the value of the selected_nominal_classifier variable)", default="default")
	argparser.add_argument('split', help="train: use first 2/3 of every author's tweets. test: use last 1/3 of every author's tweets. any other value: use all tweets", default='all')
	args = argparser.parse_args()
	return args.inputdir, args.outputdir, args.modeldir, args.targetattrs, args.featureprefixes, args.combinetweets, args.numericclassifier, args.nominalclassifier, args.split


def arff_encoder_instances(unlabeled_instance_dict, author_ids, class_attributes, include_author_ids=False, include_truth_labels=False):
	# keys of unlabeled_instance_dict are author_ids, values are lists of numpy ndarrays (where each ndarray is a 2000+ element featurized tweet)
	instances = []
	for author_id in author_ids: #sorted(authors.keys()):
		if include_truth_labels and author_id in author_truth_dict:
			labels = [author_truth_dict[author_id][attribute] for attribute in class_attributes]
		else:
			labels = ['?']

		if include_author_ids:
			instances.extend([[author_id] + list(instance_nolabel) + labels for instance_nolabel in unlabeled_instance_dict[author_id]])
		else:
			instances.extend([list(instance_nolabel) + labels for instance_nolabel in unlabeled_instance_dict[author_id]])
	return instances


def __main__():
	print("Classifier testing program running in directory", os.path.abspath(os.curdir))
	print("***", time.ctime())

	# Uncomment to get input and output directory names from command line arguments to this script
	inputdir, outputdir, modeldir, targetattrs, featureprefixes, combinetweets, numericclassifier, nominalclassifier, tweetwise_split = get_script_args()

	inputdir = os.path.join(os.curdir, inputdir)
	outputdir = os.path.join(os.curdir, outputdir)
	modeldir = os.path.join(os.curdir, modeldir)

	if os.path.exists( inputdir ) and os.path.isdir( inputdir ):
		print("Found testing inputdir", inputdir)
		if "2014" in inputdir:
			print("Found \"2014\" in inputdir name, setting config_year = \"2014\"")
			config_year = '2014'
		else:
			config_year = '2015'
	else:
		print("Couldn't find testing inputdir", inputdir)
		exit(1)
	
	if os.path.exists( outputdir ) and os.path.isdir( outputdir ):
		print("Found testing outputdir", outputdir)
	else:
		print("Couldn't find testing outputdir", outputdir)
		exit(1)
		
	if os.path.exists( modeldir ) and os.path.isdir( modeldir ):
		print("Found train/test shared modeldir", modeldir)
	else:
		print("Couldn't find train/test shared modeldir", modeldir)
		exit(1)

	tempdir = tempfile.mkdtemp(suffix='', prefix="temptest_", dir=os.curdir)
	print("Using directory", tempdir, "for temporary working files")

	if combinetweets.lower() == "n" or combinetweets.lower() == "no":
		print("Received option '-combinetweets n(o)': Do not combine multiple tweets into single instances")
		combinetweets = False
	else:
		print("Combine multiple tweets into single instances (default)")
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

	selected_feature_prefixes = all_feature_prefixes
	if featureprefixes.lower() == "all":
		print("Received option '-featureprefixes all': Using all available feature types:", selected_feature_prefixes)
	elif featureprefixes in all_feature_prefixes:
		print("Received option '-featureprefixes", featureprefixes, "': Using only features beginning with:", featureprefixes)
		selected_feature_prefixes = [featureprefixes]
	elif "." in featureprefixes:
		print("Received option '-featureprefixes {0}'".format(featureprefixes))
		featureprefixes = featureprefixes.split(".")
		print("Requested to use features beginning with one of:", featureprefixes)
		featureprefixes = [prefix for prefix in all_feature_prefixes if prefix in featureprefixes]
		if len(featureprefixes) < 1:
			print("Requested feature prefixes not in available list; using all features instead")
		else:
			selected_feature_prefixes = featureprefixes
			print("Using features beginning with one of:", selected_feature_prefixes)

	print("Requested predictions for attributes:", targetattrs)
	targetattrs = targetattrs.split(".")
	if 'all' in targetattrs:
		targetattrs = ['gender', 'age_group', 'extroverted', 'stable', 'agreeable', 'conscientious', 'open']

	input_xml_files = get_input_files(inputdir, '.xml')
	xml_filenames_by_language, userids_by_language = group_by_language(input_xml_files, config_year)
	# xml_filenames_by_language: keys are ('en', 'es', 'it', 'nl'), values are lists of .xml filenames
	# userids_by_language: keys are ('en', 'es', 'it', 'nl'), values are lists of userid strings
	
	for language in xml_filenames_by_language:
		# Write intermediate .txt files holding cleaned-up and tokenized tweet content, one tweet per line of a file
		write_preprocessed_tweets(language, xml_filenames_by_language[language], tempdir, tweetwise_split)
		
	# NO TRUTH FILES TO PROCESS FOR TESTING SET
	
	unlabeled_authors = get_unlabeled_authors(userids_by_language, tempdir, force_lowercase=False, config_year=config_year)
	# unlabeled_authors:
	#	keys are ('en', 'es', 'it', 'nl'),
	#	values are dictionaries whose keys are author userid strings, values are UnlabeledAuthor objects
	
	for testing_language in sorted(unlabeled_authors.keys()):
		testing_authors = unlabeled_authors[testing_language]
		if len(testing_authors) < 1:
			continue
		print("*** Starting {0} at {1}".format(testing_language, time.ctime()))
		
		# ------ MALLET TIME ------
		# generate a text file that mallet can import:
		# one "document" (tweet) per line. format is:
		# 	document-id[SPACE]label[SPACE]whitespace-delimited-utf8-encoded-tokens
		# document-id will be the index of that tweet in a sorted list of all tweets in this language,
		# label will be the author's userid (I guess?)
		# the last field is the tweet content, which we have already preprocessed

		alltweets = []
		num_topics = 100

		author_tweet_ranges = dict.fromkeys(testing_authors.keys()) # dictionary to keep track of the first and last tweet indexes belonging to each author
		tweets_written = 0
		for author_id in sorted(testing_authors.keys()):
			if len(testing_authors[author_id].tweets) > 0:
				author_tweet_ranges[author_id] = {'first': tweets_written, 'last': tweets_written + len(testing_authors[author_id].tweets)}
			for tweet in testing_authors[author_id].tweets:
				alltweets.append([tweets_written, author_id, tweet])
				tweets_written += 1
		alltweets_filename = os.path.join(tempdir, "all_{lang}_tweets.txt".format(lang=testing_language))
		alltweets_keys_filename = os.path.join(tempdir, "all_{lang}_tweets_keys.txt".format(lang=testing_language))
		
		with open(alltweets_filename, mode='w', encoding='utf8') as alltweets_file:
			alltweets_file.write("\n".join(["{}\t{}\t{}".format(num, label, text) for num, label, text in alltweets]))
			print("Wrote", len(alltweets), "tweets to", alltweets_filename, "for processing in MALLET")
			
		with open(alltweets_keys_filename, mode='w', encoding='utf8') as alltweets_keys_file:
			for author_id, range_data in sorted(author_tweet_ranges.items()):
				author_data = testing_authors[author_id]
				alltweets_keys_file.write( "{},{},{}\n".format(author_id, range_data['first'], range_data['last']) )
			print("Wrote first and last tweet indexes for all {lang} authors to file".format(lang=testing_language), alltweets_keys_filename)

		if sys.platform != "win32":
			mallet_files_cmd = "sh make_mallet_files_testing.sh {lang} {filedir} {num_topics}".format(lang=testing_language, filedir=tempdir, num_topics=num_topics)
			print("TEST: About to make subprocess call:", mallet_files_cmd)
			call(mallet_files_cmd, shell=True)

		if os.path.exists(os.path.join(tempdir, "{lang}_{num_topics}_corpus.mallet".format(lang=testing_language, num_topics=num_topics))):
			print("\nTEST:", os.path.join(tempdir, "{lang}_{num_topics}_corpus.mallet".format(lang=testing_language, num_topics=num_topics)), "exists!")
		else:
			print("Failed to make the file", os.path.join(tempdir, "{lang}_{num_topics}_corpus.mallet".format(lang=testing_language, num_topics=num_topics)))
			continue

		if os.path.exists(os.path.join(tempdir, "{lang}_{num_topics}_doctopics.txt".format(lang=testing_language, num_topics=num_topics))):
			print("TEST:", os.path.join(tempdir, "{lang}_{num_topics}_doctopics.txt".format(lang=testing_language, num_topics=num_topics)), "exists!\n")
		else:
			print("Failed to make the file", os.path.join(tempdir, "{lang}_{num_topics}_doctopics.txt".format(lang=testing_language, num_topics=num_topics)))
			continue

		doctopics_test(testing_language, tempdir, num_topics)
		print("\nTEST: Continuing after doctopics_test")
		# ------ END OF MALLET TIME ------
		
		if testing_language in ('it', 'nl'): # these two languages are not labeled with author's age group
			testing_attributes = ['gender', 'extroverted', 'stable', 'agreeable', 'conscientious', 'open']
		else: # 'es' and 'en' are labeled with age group
			if config_year == '2015':
				testing_attributes = ['gender', 'age_group', 'extroverted', 'stable', 'agreeable', 'conscientious', 'open']
			else:
				testing_attributes = ['gender', 'age_group']

		# From attributes that are possible to predict for this language/configuration, take only those specified as a script argument
		testing_attributes = [attribute for attribute in testing_attributes if attribute in targetattrs]

		feature_names = []
		if 'topic' in selected_feature_prefixes:
			feature_names += ["topic{0}".format(str(topic).zfill(4)) for topic in range(0,max_topic_index+1)]
		for name in custom_feature_names:
			for prefix in selected_feature_prefixes:
				if name.startswith(prefix):
					feature_names += [name]
					continue
		print("TEST: Found", len(feature_names), "feature names matching selected feature prefixes:", selected_feature_prefixes)
		print(", ".join([name for name in feature_names if not name.startswith("topic")]))
		if 'topic' in selected_feature_prefixes:
			print(len([name for name in feature_names if name.startswith("topic")]), "topics features\n")
		else:
			print("")


		weka_features = get_weka_features(None, feature_names)

		arff_enc = MyArffEncoder()
		testing_instances = extract_instance_features(testing_authors, feature_names, combinetweets)

		# Export one .arff PER LANGUAGE that has ALL THE TESTING ATTRIBUTES available for this language
		if export_combined_arff:
			print("TEST: Preparing to write file of testing instances for", len(testing_authors), "authors in", testing_language, testing_attributes)
			all_authors_testing_arff1 = ArffDict(
				relation = "{0} sorted author_ids: {1}".format(testing_language, ",".join(sorted(testing_authors.keys()))),
				attributes = [('author_id', [author_id for author_id in testing_authors])]
							 + weka_features
							 + [(attribute, labels_for_attribute[attribute]) for attribute in testing_attributes],
				data = arff_encoder_instances(testing_instances, sorted(testing_authors.keys()), testing_attributes, include_author_ids=True, include_truth_labels=True),
			)
			all_authors_filename1 = "{prefix}_allattrs_test.arff".format(prefix=testing_language)
			all_authors_filepath1 = os.path.join(tempdir, all_authors_filename1)
			with open(all_authors_filepath1, mode='w', encoding='utf8') as test_file:
				for instance in arff_enc.iter_encode(all_authors_testing_arff1):
					test_file.write(instance + u'\n')
			print("TEST: MyArffEncoder wrote", testing_language, testing_attributes, "testing data file:", all_authors_filepath1)
		
		for class_attribute in testing_attributes:
			print("*** Starting {0}, {1} at {2}".format(testing_language, class_attribute, time.ctime()))

			short_classifier_name = ""

			testing_labels = labels_for_attribute[class_attribute]
			if class_attribute in PERSONALITY_TRAITS:
				testing_labels = 'NUMERIC'

				# Get the desired classifier classname from our chosen config in weka_classify.py. It should match the classifier that we used for training
				classifier_classname = numeric_classifier_config["test_classifier_classname"]
				short_classifier_name = numeric_classifier_config["name"]
			else:
				testing_labels = sorted(testing_labels)
				# Get the desired classifier classname from our chosen config in weka_classify.py. It should match the classifier that we used for training
				classifier_classname = nominal_classifier_config["test_classifier_classname"]
				short_classifier_name = nominal_classifier_config["name"]

			language_class_prefix = "{lang}_{classattr}".format(lang=testing_language, classattr=class_attribute)

			for data_source in ("py","mallet"):
				model_filename = "weka_{lang_class}_{classifier}_{source}.model".format(lang_class=language_class_prefix, classifier=short_classifier_name, source=data_source)
				# model_filename = "weka_" + language_class_prefix + "_" + short_classifier_name + ".model"
				model_filepath = os.path.join(modeldir, model_filename)
				print('\nTEST: looking for model file', model_filepath)
				if os.path.exists( model_filepath ) and os.path.isfile( model_filepath ):
					print("Found trained model file", model_filepath)
				else:
					print("Couldn't find trained model file", model_filepath)
					print("Contents of modeldir", modeldir)
					print(os.listdir(modeldir))
					print('\nTEST: skipping attribute', class_attribute, "for language:", testing_language)
					continue

				weka_classifier_options = []
				weka_classifier = weka_classify.myweka.WekaClassifier(model_filepath, classifier_classname)


				# Export one .arff PER (LANGUAGE, CLASS_ATTRIBUTE) pair that has only the class attribute currently being predicted
				if test_combined_arff:
					if data_source == 'py':
						print("TEST: Preparing to write file of testing instances for", len(testing_authors), "authors in", language_class_prefix)
						all_authors_testing_arff = ArffDict(
							# relation = "{0} sorted author_ids: {1}".format(language_class_prefix, ",".join(sorted(testing_authors.keys()))),
							relation = language_class_prefix,
							attributes = [('author_id', [author_id for author_id in testing_authors])]
										 + weka_features
										 + [(class_attribute, testing_labels)],
							data = arff_encoder_instances(testing_instances, sorted(testing_authors.keys()), [class_attribute,], include_author_ids=True, include_truth_labels=known_labels),
						)
						all_authors_filename = "{prefix}_test_{classifier}_{source}.arff".format(prefix=language_class_prefix, classifier=short_classifier_name, source=data_source)
						all_authors_filepath = os.path.join(tempdir, all_authors_filename)
						with open(all_authors_filepath, mode='w', encoding='utf8') as test_file:
							for instance in arff_enc.iter_encode(all_authors_testing_arff):
								test_file.write(instance + u'\n')
						print("TEST: MyArffEncoder wrote", language_class_prefix, "testing data file:", all_authors_filepath)
					else:
						# the mallet arff file doesn't have the classifier in the filename because we use the same data file for numeric and nominal classes
						all_authors_filename = "{prefix}_test_{source}.arff".format(prefix=language_class_prefix, source=data_source)
						all_authors_filepath = os.path.join(tempdir, all_authors_filename)

					print("\n***", time.ctime())
					print("TEST: making predictions for all authors in", language_class_prefix)
					results = weka_classifier._classify_many(all_authors_filepath, weka_classifier_options, show_stdout=True)
					print("***", time.ctime())


					print("TEST:", len(results)-1, "predictions made by", language_class_prefix, "classifier:", classifier_classname)
					with open(all_authors_filepath.replace(".arff", ".csv"), mode='w', encoding='utf8') as test_csv_file:
						test_csv_file.write('\n'.join(results))
					print("TEST: wrote predictions for all authors in", language_class_prefix, "to file", all_authors_filepath.replace(".arff", ".csv"))

					author_predictions = dict.fromkeys(testing_authors.keys(), None)

					for result_line in results[1:]:
						#instance_number, actual_value, predicted_value, prediction_error, author_id = result_line.split() # split on any whitespace
						split_result_line = result_line.split()

						actual_value = split_result_line[1]
						if ":" in actual_value:
							actual_value = actual_value.split(":")[1]

						predicted_value = split_result_line[2]
						if predicted_value == "?":
							print("Predicted value was '?'. Result line was:\n", result_line)
							continue
						elif ":" in predicted_value:
							predicted_value = predicted_value.split(":")[1]

						if known_labels:
							is_error = 0
							if "+" in split_result_line:
								is_error = 1

						author_id = split_result_line[-1].strip('()')
						if author_predictions[author_id] is None:
							author_predictions[author_id] = {'predictions': [], 'error_count': 0, 'actual_value': None}

						if testing_labels is "NUMERIC":
							if known_labels:
								try:
									actual_value = float(actual_value)
								except:
									print("Error converting actual value to float:", actual_value, "result line was:")
									print(result_line, "\n")

							try:
								predicted_value = float(predicted_value)
							except:
								print("Error converting predicted value to float:", predicted_value, "result line was:")
								print(result_line, "\n")

						author_predictions[author_id]['predictions'].append(predicted_value)
						author_predictions[author_id]['actual_value'] = actual_value
						if known_labels:
							author_predictions[author_id]['error_count'] += is_error

					print("***********************************************")
					if testing_labels is "NUMERIC":
						print("\nshort_id\tmean\tmedian\tstdev\tactual")

					correct_authors = 0
					for author_id in sorted(author_predictions.keys()):
						prediction_dict = author_predictions[author_id]
						all_predictions_count = len(prediction_dict['predictions'])
						if testing_labels is "NUMERIC":
							prediction_mean = mean(array(prediction_dict['predictions'],dtype='float'))
							prediction_median = median(array(prediction_dict['predictions'],dtype='float'))
							prediction_std = std(array(prediction_dict['predictions'],dtype='float'))

							print("{shortid}\t{mean:4}\t{median:4}\t{stdev:4}\t{actual}".format(
								shortid=author_id[:8], mean=round(prediction_mean,4), median=round(prediction_median,4),
								stdev=round(prediction_std,4), actual=prediction_dict['actual_value']
							))
							testing_authors[author_id].make_prediction(class_attribute, prediction_median)
						else:
							label_dict = dict.fromkeys(testing_labels)
							for label in testing_labels:
								label_dict[label] = prediction_dict['predictions'].count(label)
							consensus, consensus_votes = sorted(label_dict.items(), key=itemgetter(1), reverse=True)[0]
							percent_consensus_votes = round((consensus_votes / all_predictions_count) * 100)

							label_strings = []
							for label in testing_labels:
								this_prediction_count = prediction_dict['predictions'].count(label)
								label_marking = ""
								if label == consensus:
									label_marking += "*"
								if label == prediction_dict['actual_value']:
									label_marking += "**"
								label_strings.append("{label}{mark:3}  {percentage:3}%".format(label=label, mark=label_marking, percentage=round((this_prediction_count / all_predictions_count)*100)))

							# percent_correct_votes = round((all_predictions_count-prediction_dict['error_count'] / all_predictions_count) * 100)

							consensus_correct = ""
							if consensus == prediction_dict['actual_value']:
								consensus_correct = "\tCORRECT"
								correct_authors += 1

							print("{shortid}  {label_strings} {consenus_correct}".format(
								shortid=author_id[:8], label_strings="    ".join(label_strings), consenus_correct=consensus_correct))

							testing_authors[author_id].make_prediction(class_attribute, consensus)

					if testing_labels is not "NUMERIC" and known_labels:
						percent_correct_authors = round((correct_authors / len(testing_authors)) * 100, 1)
						print("Correctly classified authors: {correct_percent}% of {total}\n".format(correct_percent=percent_correct_authors, total=len(testing_authors)))

		print("***", time.ctime())
		print("Writing author predicton .xml files to", outputdir)
		for author_id in sorted(testing_authors.keys()):
			author_data = testing_authors[author_id]
			author_output_filepath = os.path.join(outputdir, author_id + ".xml")
			with open(author_output_filepath, mode='w', encoding='utf8') as author_output_file:
				author_output_file.write(author_data.get_prediction_xml())

	print("***", time.ctime())
	# print('Removing temporary directory', tempdir)
	print("Cleaning up .txt files from temporary directory", tempdir)
	for f in os.listdir(tempdir):
		if f.endswith('.txt') and ('tweets' not in f) and ('doctopics' not in f):
			os.remove(os.path.join(tempdir, f))
	# os.rmdir(tempdir)
__main__()
