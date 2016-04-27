import os, argparse
from xml.etree.ElementTree import ElementTree
from math import sqrt

LANGUAGES = ('en', 'es', 'it', 'nl')
GENDERS = ('M', 'F')
AGE_GROUPS = ('18-24', '25-34', '35-49', '50-XX')
PERSONALITY_TRAITS = ('extroverted', 'stable', 'agreeable', 'conscientious', 'open')

def get_directory_args():
	argparser = argparse.ArgumentParser(description = "Compare output of Author Profiling classification with known labels and calculate RMSE per attribute")
	argparser.add_argument('--truthdir', dest='truth_input_dir', help="path to directory containing actual author attribute data, one XML file per author", default="")
	argparser.add_argument('--predictiondir', dest='prediction_output_dir', help="path to directory containing predicted author attribute data, one XML file per author", default="")
	argparser.add_argument('-v', dest='verbose', const=True, default=False, nargs='?')
	args = argparser.parse_args()
	return args.truth_input_dir, args.prediction_output_dir, args.verbose

def get_input_files(inputdir, extension):
	filenames = []
	for root, dirs, files in os.walk(inputdir):
		filenames.extend( [os.path.join(root, file) for file in files if file.endswith(extension)] )
	print("Found", len(filenames), extension, "files in", inputdir)
	return filenames

def find_xml_file(filename_list, basename):
	found_file = [file for file in filename_list if os.path.basename(file) == basename]
	if len(found_file) == 0:
		return None
	else:
		return found_file[0]

def get_attribute_error(attribute_name, value1, value2):
	if value1 == value2:
		return 0
	elif attribute_name in PERSONALITY_TRAITS:
		# return abs( float(value1) - float(value2) )
		return float(value1) - float(value2)
	elif attribute_name == "gender":
		return 1 # not sure how to scale this, so for now just give an error of 1 if they don't match
	elif attribute_name == "age_group":
		return abs( AGE_GROUPS.index(value1) - AGE_GROUPS.index(value2) ) / 3 # divide by maximum distance so error will be between 0 and 1 ?

def __main__():
	truth_input_dir, prediction_output_dir, verbose = get_directory_args()
	
	print("PREDICTION FILES:")
	if prediction_output_dir == "":
		prediction_output_dir = os.path.join(os.curdir, 'output', 'test')
	prediction_file_names = get_input_files(prediction_output_dir, ".xml")

	print("TRUTH FILES:")
	if truth_input_dir == "":
		truth_input_dir = os.path.join(os.curdir, 'include', 'training_set_predictions')
	truth_file_names = get_input_files(truth_input_dir, ".xml")

	prediction_error = {}
	prediction_correct = {}
	prediction_incorrect = {}
	for attribute_name in ('age_group', 'gender', 'extroverted', 'stable', 'agreeable', 'conscientious', 'open'):
		prediction_error[attribute_name] = []
		prediction_correct[attribute_name] = []
		prediction_incorrect[attribute_name] = []

	instance_count = 0
	instance_count_with_age = 0

	for prediction_filename in prediction_file_names:
		if verbose:
			print("\nPREDICTION ", prediction_filename)
		xmldoc_prediction = ElementTree(file = prediction_filename)
		author_prediction = xmldoc_prediction.getroot().find('.')

		truth_filename = find_xml_file(truth_file_names, os.path.basename(prediction_filename))
		if verbose:
			print("TRUTH ", truth_filename)
		xmldoc_truth = ElementTree(file = truth_filename) # <xml.etree.ElementTree.ElementTree object at 0x02F89630>
		author_truth = xmldoc_truth.getroot().find('.') # <Element 'author' at 0x02F8D390>

		instance_count += 1
		if author_prediction.get('lang') in ('en', 'es'): # nl and it have all ages labeled XX-XX so don't count in mean
			instance_count_with_age += 1

		for attribute_name in ('id', 'type', 'lang'):
			if author_prediction.get(attribute_name) != author_truth.get(attribute_name):
				print(author_prediction.get(attribute_name), "!=", author_truth.get(attribute_name))
				print("Error:", attribute_name, "values did not match between truth and prediction file")
				return
		
		for attribute_name in ('age_group', 'gender', 'extroverted', 'stable', 'agreeable', 'conscientious', 'open'):
			if author_prediction.get(attribute_name) != author_truth.get(attribute_name):
				this_attribute_error = get_attribute_error(attribute_name, author_prediction.get(attribute_name), author_truth.get(attribute_name))
				prediction_error[attribute_name] += [pow(this_attribute_error,2)]
				if this_attribute_error > 0.1:
					prediction_incorrect[attribute_name].append("{0}_{1}_{2}".format(author_prediction.get('lang'), author_prediction.get('id'), attribute_name))
				else:
					prediction_correct[attribute_name].append("{0}_{1}_{2}".format(author_prediction.get('lang'), author_prediction.get('id'), attribute_name))
				if verbose:
					print("[{lang} {userid}] {attname:13}(prediction): {predvalue:5}  {attname:13}(truth): {truthvalue:5}  error: {err}".format(
						userid = author_prediction.get('id'),
						lang = author_prediction.get('lang'),
						attname = attribute_name,
						predvalue = author_prediction.get(attribute_name),
						truthvalue = author_truth.get(attribute_name),
						err = "%1.2f" % abs(get_attribute_error(attribute_name, author_prediction.get(attribute_name), author_truth.get(attribute_name))),
						))
			else:
				prediction_correct[attribute_name].append("{0}_{1}_{2}".format(author_prediction.get('lang'), author_prediction.get('id'), attribute_name))

	print("\nProcessed", instance_count, "instances\n")
	personality_rmse_values = []
	for attribute_name in ('age_group', 'gender', 'extroverted', 'stable', 'agreeable', 'conscientious', 'open'):
		if attribute_name == 'age_group':
			count = instance_count_with_age
		else:
			count = instance_count

		if count > 0:
			print("RMSE in predictions for {attname:13}: {err}, ({correctcount}/{total}\tauthors == {percent}% \"correct\" -- abs error <= 0.1)".format(
				attname = attribute_name,
				err = "%1.3f" % sqrt( sum(prediction_error[attribute_name]) / count ),
				correctcount = len(prediction_correct[attribute_name]),
				total = count,
				percent = "%1.1f" % ((float(len(prediction_correct[attribute_name])) / count) * 100),
				))
			if attribute_name in PERSONALITY_TRAITS:
				personality_rmse_values.append(sqrt( sum(prediction_error[attribute_name]) / count ))

	if len(personality_rmse_values) > 0:
		print("RMSE in predictions for {attname:13}: {err}".format(attname="personality", err="%1.3f" % (sum(personality_rmse_values)/len(personality_rmse_values))))
__main__()