import os

from nltk.internals import java, config_java

import environment_settings

# import nltk.classify

# I'm using myutil in place of nltk.classify.util so I can customize the code for my purposes
# import nltk.classify.util # for accuracy & log_likelihood



if 'JAVAHOME' not in os.environ:
	os.environ['JAVAHOME'] = os.path.abspath( environment_settings.javahome )

# javaw_path = os.path.abspath(os.path.join(os.environ['JAVAHOME'], 'javaw.exe'))
# config_java(bin=javaw_path)

if environment_settings.javaw_path is not None and os.path.exists(environment_settings.javaw_path):
	config_java(bin = environment_settings.javaw_path)
else:
	config_java()

if 'WEKAHOME' not in os.environ:
	os.environ['WEKAHOME'] = os.path.abspath( environment_settings.wekahome )


import myweka # my modifications made directly to a local copy of the nltk.classify.weka source code
import myutil # my modifications made directly to a local copy of the nltk.classify.util source code
myweka.config_weka(classpath = os.environ['WEKAHOME'])

numeric_classifier_configs = {
	"RandomSubSpace1": {
		"name": "RandomSubSpace1",
		"multifilter_cmd": 'weka.filters.MultiFilter -F "weka.filters.unsupervised.attribute.Remove -R first" -F "weka.filters.unsupervised.instance.NonSparseToSparse"',
		"train_cmd_format": "weka.classifiers.meta.RandomSubSpace -d {output_model_file} -t {filtered_arff_file} -P 0.15 -S 1 -num-slots 0 -I 20 {crossval} -W weka.classifiers.trees.REPTree -- -L 10",
		"test_classifier_classname": "weka.classifiers.meta.RandomSubSpace",
	},
	"ZeroR1": {
		"name": "ZeroR1",
		"multifilter_cmd": 'weka.filters.MultiFilter -F "weka.filters.unsupervised.attribute.Remove -R first" -F "weka.filters.unsupervised.instance.NonSparseToSparse"',
		"train_cmd_format": "weka.classifiers.rules.ZeroR -d {output_model_file} -t {filtered_arff_file} {crossval}",
		"test_classifier_classname": "weka.classifiers.rules.ZeroR",
	},
	"FilteredREPTree" : {
		"name": "FilteredREPTree",
		"multifilter_cmd": '',
		"train_cmd_format": 'weka.classifiers.meta.FilteredClassifier -d {output_model_file} -t {filtered_arff_file} -F "weka.filters.unsupervised.attribute.Remove -R first" {crossval} -W weka.classifiers.trees.REPTree -- -L 10',
		"test_classifier_classname": "weka.classifiers.trees.REPTree",
	},
	"FilteredGP" : {
		"name": "FilteredGP",
		"multifilter_cmd": '',
		"train_cmd_format": 'weka.classifiers.meta.FilteredClassifier -d {output_model_file} -t {filtered_arff_file} -F "weka.filters.unsupervised.attribute.Remove -R first" {crossval} -W weka.classifiers.functions.GaussianProcesses -- -L 1.1 -N 0 -K "weka.classifiers.functions.supportVector.Puk -O 1.5 -S 0.7"',
		"test_classifier_classname": "weka.classifiers.functions.GaussianProcesses",
	},
	"FilteredZeroR" : {
		"name": "FilteredZeroR",
		"multifilter_cmd": '',
		"train_cmd_format": 'weka.classifiers.meta.FilteredClassifier -d {output_model_file} -t {filtered_arff_file} -F "weka.filters.unsupervised.attribute.Remove -R first" {crossval} -W weka.classifiers.rules.ZeroR',
		"test_classifier_classname": "weka.classifiers.rules.ZeroR",
	},
	# weka.classifiers.meta.FilteredClassifier -F "weka.filters.MultiFilter -F \"weka.filters.unsupervised.attribute.Remove -R 1\" -F \"weka.filters.supervised.attribute.AttributeSelection -E \\\"weka.attributeSelection.CfsSubsetEval -P 1 -E 1\\\" -S \\\"weka.attributeSelection.GreedyStepwise -T 0.0 -N -1 -num-slots 0\\\"\"" -W weka.classifiers.meta.Bagging -- -P 50 -S 1 -num-slots 0 -I 20 -W weka.classifiers.trees.REPTree -- -M 2 -V 0.01 -N 5 -S 1 -L 15 -I 0.0
	"FilteredBagging" : {
		"name": "FilteredBagging",
		"multifilter_cmd": '',
		# "train_cmd_format": 'weka.classifiers.meta.FilteredClassifier -d {output_model_file} -t {filtered_arff_file} -F "weka.filters.MultiFilter -F \"weka.filters.unsupervised.attribute.Remove -R 1\" -F \"weka.filters.supervised.attribute.AttributeSelection -E \\\"weka.attributeSelection.CfsSubsetEval -P 1 -E 1\\\" -S \\\"weka.attributeSelection.GreedyStepwise -T 0.0 -N -1 -num-slots 0\\\"\"" {crossval} -W weka.classifiers.trees.REPTree -- -L 5',
		# weka.classifiers.meta.FilteredClassifier -F "weka.filters.MultiFilter -F \"weka.filters.unsupervised.attribute.Remove -R 1\"" -W weka.classifiers.meta.Bagging -- -P 50 -S 1 -num-slots 4 -I 20 -W weka.classifiers.trees.M5P -- -M 4.0

		# The -o in next line, directly after the crossval spec, tells weka not to output THE ENTIRE MODEL to stdout, just the summary statistics
		"train_cmd_format": 'weka.classifiers.meta.FilteredClassifier -d {output_model_file} -t {filtered_arff_file} -F "weka.filters.MultiFilter -F \\"weka.filters.unsupervised.attribute.Remove -R 1\\" -F \\"weka.filters.supervised.attribute.AttributeSelection -E \\\\\\"weka.attributeSelection.CfsSubsetEval -P 1 -E 1\\\\\\" -S \\\\\\"weka.attributeSelection.GreedyStepwise -T 0.0 -N -1 -num-slots 0\\\\\\"\\"" {crossval} -o -W weka.classifiers.meta.Bagging -- -P 50 -S 1 -num-slots 0 -I 20 -W weka.classifiers.trees.REPTree -- -M 2 -V 0.01 -N 5 -S 1 -L 15 -I 0.0',
		"test_classifier_classname": "weka.classifiers.meta.Bagging",
	},
}
# ---- SET THE DESIRED CLASSIFIER FOR NUMERIC ATTRIBUTES HERE ----
selected_numeric_classifier = "FilteredBagging" # should be one of the keys of numeric_classifier_configs above

# the following two variables will be checked from setup_training:
numeric_multifilter_cmd = numeric_classifier_configs[selected_numeric_classifier]["multifilter_cmd"]
numeric_train_cmd_format = numeric_classifier_configs[selected_numeric_classifier]["train_cmd_format"]

# this variable will be checked from setup_testing:
numeric_test_classifier_classname = numeric_classifier_configs[selected_numeric_classifier]["test_classifier_classname"]

nominal_classifier_configs = {
	# other_filter_cmd is:
	# 	filter_cmd_format = "weka.filters.unsupervised.attribute.AddValues -C last -S -L {labels}"
	# 	filter_cmd = filter_cmd_format.format(labels=",".join(training_labels))
	"LibLINEAR1": {
		"name": "LibLINEAR1",
		"multifilter_cmd": 'weka.filters.MultiFilter -F "{other_filter_cmd}" -F "weka.filters.unsupervised.attribute.Remove -R first" -F "weka.filters.unsupervised.instance.NonSparseToSparse"',
		"train_cmd_format": "weka.Run weka.classifiers.functions.LibLINEAR -d {output_model_file} -t {filtered_arff_file} -S 1 -C 1.0 -E 0.001 -B 1.0 {crossval}",
		"test_classifier_classname": "weka.classifiers.functions.LibLINEAR",
	},
	"ZeroR1": {
		"name": "ZeroR1",
		"multifilter_cmd": 'weka.filters.MultiFilter -F "{other_filter_cmd}" -F "weka.filters.unsupervised.attribute.Remove -R first" -F "weka.filters.unsupervised.instance.NonSparseToSparse"',
		"train_cmd_format": "weka.classifiers.rules.ZeroR -d {output_model_file} -t {filtered_arff_file} {crossval}",
		"test_classifier_classname": "weka.classifiers.rules.ZeroR",
	},
	# mccollister15@tira-ubuntu:~$ java -cp /home/mccollister15/weka-3-7-12/weka.jar weka.classifiers.meta.Bagging -P 100 -S 1
	#  -num-slots 2 -I 10 -W weka.classifiers.trees.J48
	#   -d en_age_bagging.model -t output_fbtopics/train/en_age_group_train_filtered.arff -x 10
	"BaggingJ48": {
		"name": "BaggingJ48",
		"multifilter_cmd": 'weka.filters.MultiFilter -F "{other_filter_cmd}" -F "weka.filters.unsupervised.attribute.Remove -R first" -F "weka.filters.unsupervised.instance.NonSparseToSparse"',
		"train_cmd_format": "weka.classifiers.meta.Bagging -P 100 -S 1 -num-slots 2 -I 10 -W weka.classifiers.trees.J48 -d {output_model_file} -t {filtered_arff_file} {crossval}",
		"test_classifier_classname": "weka.classifiers.meta.Bagging",
	},
	"FilteredLibLINEAR1": {
		"name": "FilteredLibLINEAR1",
		"multifilter_cmd": 'weka.filters.MultiFilter -F "{other_filter_cmd}" -F "weka.filters.unsupervised.instance.NonSparseToSparse"',
		"train_cmd_format": 'weka.Run weka.classifiers.meta.FilteredClassifier -d {output_model_file} -t {filtered_arff_file} -F "weka.filters.unsupervised.attribute.Remove -R first" {crossval} -W weka.classifiers.functions.LibLINEAR -- -S 1 -C 1.0 -E 0.001 -B 1.0',
		"test_classifier_classname": "weka.classifiers.functions.LibLINEAR",
	},
	"FilteredZeroR": {
		"name": "FilteredZeroR",
		"multifilter_cmd": 'weka.filters.MultiFilter -F "{other_filter_cmd}" -F "weka.filters.unsupervised.instance.NonSparseToSparse"',
		"train_cmd_format": 'weka.Run weka.classifiers.meta.FilteredClassifier -d {output_model_file} -t {filtered_arff_file} -F "weka.filters.unsupervised.attribute.Remove -R first" {crossval} -W weka.classifiers.rules.ZeroR',
		"test_classifier_classname": "weka.classifiers.rules.ZeroR",
	},
	"FilteredLibLINEAR2": {
		"name": "FilteredLibLINEAR2",
		"multifilter_cmd": 'weka.filters.MultiFilter -F "{other_filter_cmd}" -F "weka.filters.unsupervised.instance.NonSparseToSparse"',
		"train_cmd_format": 'weka.Run weka.classifiers.meta.FilteredClassifier -d {output_model_file} -t {filtered_arff_file} -F "weka.filters.unsupervised.attribute.Remove -R first" {crossval} -W weka.classifiers.functions.LibLINEAR -- -S 1 -C 0.8 -E 0.001 -B 1.0',
		"test_classifier_classname": "weka.classifiers.functions.LibLINEAR",
	},
	# weka.classifiers.meta.FilteredClassifier -F "weka.filters.MultiFilter -F \"weka.filters.unsupervised.attribute.Remove -R 1\" -F \"weka.filters.supervised.attribute.AttributeSelection -E \\\"weka.attributeSelection.CfsSubsetEval -P 1 -E 1\\\" -S \\\"weka.attributeSelection.BestFirst -D 1 -N 5\\\"\"" -W weka.classifiers.meta.RotationForest -- -G 3 -H 3 -P 80 -F "weka.filters.unsupervised.attribute.PrincipalComponents -R 1.0 -A 5 -M -1" -S 1 -num-slots 0 -I 10 -W weka.classifiers.trees.REPTree -- -M 2 -V 0.001 -N 3 -S 1 -L -1 -I 0.0
	"FilteredRotationForest": {
		"name": "FilteredRotationForest",
		"multifilter_cmd": 'weka.filters.MultiFilter -F "{other_filter_cmd}" -F "weka.filters.unsupervised.instance.NonSparseToSparse"',
		# The -o in next line, directly after the crossval spec, tells weka not to output THE ENTIRE MODEL to stdout, just the summary statistics
		"train_cmd_format": 'weka.Run weka.classifiers.meta.FilteredClassifier -d {output_model_file} -t {filtered_arff_file} -F "weka.filters.MultiFilter -F \\"weka.filters.unsupervised.attribute.Remove -R 1\\" -F \\"weka.filters.supervised.attribute.AttributeSelection -E \\\\\\"weka.attributeSelection.CfsSubsetEval -P 1 -E 1\\\\\\" -S \\\\\\"weka.attributeSelection.BestFirst -D 1 -N 5\\\\\\"\\""  {crossval} -o -W weka.classifiers.meta.RotationForest -- -G 3 -H 3 -P 80 -F "weka.filters.unsupervised.attribute.PrincipalComponents -R 1.0 -A 5 -M -1" -S 1 -num-slots 0 -I 10 -W weka.classifiers.trees.REPTree -- -M 2 -V 0.001 -N 3 -S 1 -L -1 -I 0.0',
		"test_classifier_classname": "weka.classifiers.meta.RotationForest",
	},
}
# ---- SET THE DESIRED CLASSIFIER FOR NOMINAL ATTRIBUTES HERE ----
selected_nominal_classifier = "FilteredRotationForest" # should be one of the keys of nominal_classifier_configs above

# the following two variables will be checked from setup_training:
nominal_multifilter_cmd = nominal_classifier_configs[selected_nominal_classifier]["multifilter_cmd"]
nominal_train_cmd_format = nominal_classifier_configs[selected_nominal_classifier]["train_cmd_format"]

# this variable will be checked from setup_testing:
nominal_test_classifier_classname = nominal_classifier_configs[selected_nominal_classifier]["test_classifier_classname"]
