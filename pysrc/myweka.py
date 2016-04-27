# Natural Language Toolkit: Interface to Weka Classsifiers
#
# Copyright (C) 2001-2014 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

"""
Classifiers that make use of the external 'Weka' package.
"""
from __future__ import print_function
import time
import tempfile
import os
import subprocess
import re
import zipfile

from sys import stdin
from nltk import compat
from nltk.probability import DictionaryProbDist
from nltk.internals import java, config_java

from nltk.classify.api import ClassifierI

_weka_classpath = None
_weka_search = ['.',
				'/usr/share/weka',
				'/usr/local/share/weka',
				'/usr/lib/weka',
				'/usr/local/lib/weka',]
def config_weka(classpath=None):
	global _weka_classpath

	# Make sure java's configured first.
	config_java()

	if classpath is not None:
		_weka_classpath = classpath

	if _weka_classpath is None:
		searchpath = _weka_search
		if 'WEKAHOME' in os.environ:
			searchpath.insert(0, os.environ['WEKAHOME'])

		for path in searchpath:
			if os.path.exists(os.path.join(path, 'weka.jar')):
				_weka_classpath = os.path.join(path, 'weka.jar')
				version = _check_weka_version(_weka_classpath)
				if version:
					print(('[Found Weka: %s (version %s)]' %
						   (_weka_classpath, version)))
				else:
					print('[Found Weka: %s]' % _weka_classpath)
				_check_weka_version(_weka_classpath)

	if _weka_classpath is None:
		raise LookupError('Unable to find weka.jar!  Use config_weka() '
						  'or set the WEKAHOME environment variable. '
						  'For more information about Weka, please see '
						  'http://www.cs.waikato.ac.nz/ml/weka/')

def _check_weka_version(jar):
	try:
		zf = zipfile.ZipFile(jar)
	except SystemExit as KeyboardInterrupt:
		raise
	except:
		return None
	try:
		try:
			return zf.read('weka/core/version.txt')
		except KeyError:
			return None
	finally:
		zf.close()

class WekaClassifier(ClassifierI):
	#def __init__(self, formatter, model_filename, classifier='weka.classifiers.bayes.NaiveBayes', fileprefix=''):
	def __init__(self, model_filename, classifier_classname, fileprefix='', formatter=None):
		self._formatter = formatter
		self._model = model_filename
		self.classifier = classifier_classname
		self.fileprefix = fileprefix

	def prob_classify_many(self, featuresets):
		return self._classify_many(featuresets, ['-p', '0', '-distribution'])

	def classify_many(self, featuresets):
		return self._classify_many(featuresets, ['-p', '0'])
	
	def _classify_many(self, test_instances_filepath, options, show_stdout=False):
		# Make sure we can find java & weka.
		# config_weka()

		try:
			# We have already written the test instances to a file, whose name was passed in as test_instances_filepath

			# Example (WORKING) cmd for InputMappedClassifier, where the loaded model is a FilteredClassifier
			# weka.classifiers.misc.InputMappedClassifier -I -M -L .\shared\weka_en_conscientious_Filtered.model -T en_test_1094.arff -t en_test_1094.arff -classifications "weka.classifiers.evaluation.output.prediction.CSV –p 1"
			prediction_filepath = test_instances_filepath[:-5] + ".csv"
			options = '-classifications "weka.classifiers.evaluation.output.prediction.CSV –p 1 -file {path}"'.format(path=prediction_filepath)
			options = options.split(' ')
			## INPUTMAPPEDCLASSIFIER
			cmd = ["weka.classifiers.misc.InputMappedClassifier", '-I', '-M', '-L', self._model, '-T', test_instances_filepath, '-t', test_instances_filepath, '-p', '1']
			# if len(options) > 0:
			# 	cmd = cmd + options

			"""
			# Call weka to classify the data.
			if len(options) > 0:
				cmd = [self.classifier, '-l', self._model, '-T', test_instances_filepath, '-p', '0'] + options
			else:
				cmd = [self.classifier, '-l', self._model, '-T', test_instances_filepath, '-p', '0']"""
				
			if self.classifier in ('weka.classifiers.functions.LibLINEAR', 'weka.classifiers.meta.RotationForest'):
				cmd = ['weka.Run',] + cmd

			# print(cmd)
			print(" ".join(cmd))
			
			(stdout, stderr) = java(cmd, classpath=_weka_classpath,
									stdout=subprocess.PIPE,
									stderr=subprocess.PIPE)

			# Check if something went wrong:
			if stderr and not stdout:
				print("STDERR was:")
				for line in stderr.decode(stdin.encoding).split('\n'):
					print(line)
				# if 'Illegal options: -distribution' in stderr:
					# raise ValueError('The installed version of weka does '
									 # 'not support probability distribution '
									 # 'output.\nSTDERR:\n%s' % stderr)
				# else:
					# raise ValueError('Weka failed to generate output:\n%s'
									 # % stderr)
				return None

			stdout_lines = stdout.decode(stdin.encoding).split('\n')
			if show_stdout:
				if len(stdout_lines) <= 20:
					for line in stdout_lines:
						print(line)
				else:
					print("\nFirst 20 lines of stdout were:")
					for line in stdout_lines[:20]:
						print(line)
					print("  ({0} more lines...)".format(len(stdout_lines)-20))


			# If we are using the -classifications argument with an output prediction class like weka.classifiers.evaluation.output.prediction.CSV
			# and sending prediction output to a file rather than stdout, don't bother parsing the input here
			if '-classifications' in options and '-file' in options:
				# Strip unwanted text from stdout
				for i,line in enumerate(stdout_lines):
					if line.strip().startswith("inst#"):
						stdout_lines = stdout_lines[i:]
						break
				return [line.strip() for line in stdout_lines if len(line.strip())>0 and line[0] != "="]


			# Parse weka's output.
			try:
				output_text = self.parse_weka_output(stdout_lines)
			except ValueError:
				output_text = stdout_lines
			#return self.parse_weka_output(stdout.decode(stdin.encoding).split('\n'))
			return output_text

		finally:
			#for f in os.listdir(temp_dir):
			#	os.remove(os.path.join(temp_dir, f))
			#os.rmdir(temp_dir)
			pass

	def parse_weka_distribution(self, s):
		probs = [float(v) for v in re.split('[*,]+', s) if v.strip()]
		probs = dict(zip(self._formatter.labels(), probs))
		return DictionaryProbDist(probs)

	def parse_weka_output(self, lines):
		# Strip unwanted text from stdout
		for i,line in enumerate(lines):
			if line.strip().startswith("inst#"):
				lines = lines[i:]
				break

		# print("parsing", len(lines), "lines, line 0:\n", lines[0])

		#### with CSV output:
		# inst#,actual,predicted,error,author_id
		# 1,?,0.176,?,1094e809-1306-4cd1-8ac1-ebfdf7c872b8


		#if lines[0].split()[0:4] == ['inst#', 'actual', 'predicted', 'error']: # added to handle numeric classes with no "prediction" or "distribution" column
		if ('prediction' not in lines[0]) and ('distribution' not in lines[0]):
			return [line.split()[2]
					for line in lines[1:] if line.strip()]
		#elif lines[0].split()[0:5] == ['inst#', 'actual', 'predicted', 'error', 'prediction']:
		elif ('prediction' in lines[0]) and ('distribution' not in lines[0]):
			if ":" in lines[1].split()[2]:
				return [line.split()[2].split(':')[1] for line in lines[1:] if line.strip()]
			else:
				return [line.split()[2] for line in lines[1:] if line.strip()]
		#elif lines[0].split() == ['inst#', 'actual', 'predicted', 'error', 'distribution']:
		elif ('prediction' not in lines[0]) and ('distribution' in lines[0]):
			return [self.parse_weka_distribution(line.split()[-1])
					for line in lines[1:] if line.strip()]

		# is this safe:?
		elif re.match(r'^0 \w+ [01]\.[0-9]* \?\s*$', lines[0]):
			return [line.split()[1] for line in lines if line.strip()]

		else:
			print('Unhandled output format!')
			for line in lines[:10]: print(line)
			raise ValueError('Unhandled output format -- your version '
							 'of weka may not be supported.\n'
							 '  Header: %s' % lines[0])


	# [xx] full list of classifiers (some may be abstract?):
	# ADTree, AODE, BayesNet, ComplementNaiveBayes, ConjunctiveRule,
	# DecisionStump, DecisionTable, HyperPipes, IB1, IBk, Id3, J48,
	# JRip, KStar, LBR, LeastMedSq, LinearRegression, LMT, Logistic,
	# LogisticBase, M5Base, MultilayerPerceptron,
	# MultipleClassifiersCombiner, NaiveBayes, NaiveBayesMultinomial,
	# NaiveBayesSimple, NBTree, NNge, OneR, PaceRegression, PART,
	# PreConstructedLinearModel, Prism, RandomForest,
	# RandomizableClassifier, RandomTree, RBFNetwork, REPTree, Ridor,
	# RuleNode, SimpleLinearRegression, SimpleLogistic,
	# SingleClassifierEnhancer, SMO, SMOreg, UserClassifier, VFI,
	# VotedPerceptron, Winnow, ZeroR

	_CLASSIFIER_CLASS = {
		'naivebayes': 'weka.classifiers.bayes.NaiveBayes',
		'C4.5': 'weka.classifiers.trees.J48',
		'log_regression': 'weka.classifiers.functions.Logistic',
		'svm': 'weka.classifiers.functions.SMO',
		'kstar': 'weka.classifiers.lazy.KStar',
		'ripper': 'weka.classifiers.rules.JRip',
		'adaboost': 'weka.classifiers.meta.AdaBoostM1',
		'filtered': 'weka.classifiers.meta.FilteredClassifier',
		'mnbtext': 'weka.classifiers.bayes.NaiveBayesMultinomialText',
		}
	@classmethod
	def train(cls, formatter, model_filename, featuresets, all_labels,
			  classifier='', options=[], quiet=True, fileprefix=''):
		# Make sure we can find java & weka.
		config_weka()

		#temp_dir = tempfile.mkdtemp()
		try:
			# Write the training data file.
			train_filename = os.path.join(os.curdir, fileprefix + 'train.arff')
			formatter.write(train_filename, featuresets)
			#print("\nWekaClassifier.train wrote training data file:", train_filename)
			
			# MINE MINE MINE
			if classifier == 'filtered':
				#print('requested filtered classifier, will filter and use mnbtext')
				classifier = 'mnbtext'
				# Run a weka filter on the training data set first
				filter_output = train_filename[:-5] + "_filtered.arff"
				
				filter_class = "weka.filters.unsupervised.attribute.AddValues"
				filter_cmd = [filter_class, '-C', 'last', '-S', '-L', ",".join(all_labels),
					'-i', train_filename, # input to the filter
					'-o', filter_output]
				#print("\nWekaClassifier.train filter_cmd is:", filter_cmd)
				#print(r" ".join(filter_cmd))
				
				train_filename = filter_output # use the output of the filter as the input to the classifier
				
				if quiet: stdout = subprocess.PIPE
				else: stdout = None
				java(filter_cmd, classpath=_weka_classpath, stdout=stdout)

				if classifier in cls._CLASSIFIER_CLASS:
					javaclass = cls._CLASSIFIER_CLASS[classifier]
				elif classifier in cls._CLASSIFIER_CLASS.values():
					javaclass = classifier
				else:
					raise ValueError('Unknown classifier %s' % classifier)

				# Train the weka model.
				options = []
				cmd = [javaclass,
					'-d', model_filename, # Sets model output file.
					'-t', train_filename # Sets training input data file
					]
				cmd += list(options)
				
				#print("\nWekaClassifier.train cmd is:\n", cmd, "\n")
				if quiet: stdout = subprocess.PIPE
				else: stdout = None
				java(cmd, classpath=_weka_classpath, stdout=stdout)

				# Return the new classifier.
				return WekaClassifier(formatter, model_filename,
							classifier=javaclass # I added this parameter so the _classify_many function will call the same class of classifier as we trained the model with (otherwise it always used NaiveBayes for some reason)
				)
			else:
				#print('requested classifier', classifier)
				if classifier in cls._CLASSIFIER_CLASS:
					javaclass = cls._CLASSIFIER_CLASS[classifier]
				elif classifier in cls._CLASSIFIER_CLASS.values():
					javaclass = classifier
				else:
					raise ValueError('Unknown classifier %s' % classifier)

				# Train the weka model.
				cmd = [javaclass,
					'-d', model_filename, # Sets model output file.
					'-t', train_filename # Sets training input data file
					]
				cmd += list(options)
				
				#print("\nWekaClassifier.train cmd is:\n", " ".join(cmd), "\n")
				if quiet: stdout = subprocess.PIPE
				else: stdout = None
				java(cmd, classpath=_weka_classpath, stdout=stdout)

				# Return the new classifier.
				return WekaClassifier(formatter, model_filename,
							classifier=javaclass, # I added this parameter so the _classify_many function will call the same class of classifier as we trained the model with (otherwise it always used NaiveBayes for some reason)
							fileprefix=fileprefix
				)
		finally:
			# for f in os.listdir(temp_dir):
				# os.remove(os.path.join(temp_dir, f))
			# os.rmdir(temp_dir)
			pass


class ARFF_Formatter:
	"""
	Converts featuresets and labeled featuresets to ARFF-formatted
	strings, appropriate for input into Weka.

	Features and classes can be specified manually in the constructor, or may
	be determined from data using ``from_train``.
	"""

	def __init__(self, labels, features):
		"""
		:param labels: A list of all class labels that can be generated.
		:param features: A list of feature specifications, where
			each feature specification is a tuple (fname, ftype);
			and ftype is an ARFF type string such as NUMERIC or
			STRING.
		"""
		self._labels = labels
		self._features = features

	def format(self, tokens):
		"""Returns a string representation of ARFF output for the given data."""
		return self.header_section() + self.data_section(tokens)

	def labels(self):
		"""Returns the list of classes."""
		return list(self._labels)

	def write(self, outfile, tokens):
		"""Writes ARFF data to a file for the given data."""
		if not hasattr(outfile, 'write'):
			outfile = open(outfile, 'w')
		#outfile.write(self.format(tokens))
		outfile.write( self.header_section() )
		self.data_section(outfile, tokens)
		outfile.close()

	@staticmethod
	def from_train(tokens):
		"""
		Constructs an ARFF_Formatter instance with class labels and feature
		types determined from the given data. Handles boolean, numeric and
		string (note: not nominal) types.
		"""
		# Find the set of all attested labels.
		labels = set(label for (tok,label) in tokens)

		# Determine the types of all features.
		features = {}
		for tok, label in tokens:
			for (fname, fval) in tok.items():
				if issubclass(type(fval), bool):
					ftype = '{True, False}'
				elif issubclass(type(fval), (compat.integer_types, float, bool)):
					ftype = 'NUMERIC'
				elif issubclass(type(fval), compat.string_types):
					ftype = 'STRING'
				elif fval is None:
					continue # can't tell the type.
				else:
					raise ValueError('Unsupported value type %r' % ftype)

				if features.get(fname, ftype) != ftype:
					raise ValueError('Inconsistent type for %s' % fname)
				features[fname] = ftype
		features = sorted(features.items())

		return ARFF_Formatter(labels, features)

	def header_section(self):
		"""Returns an ARFF header as a string."""
		#print("arff.write header_section info")
		# Header comment.
		s = ('% Weka ARFF file\n' +
			 '% Generated automatically by NLTK\n' +
			 '%% %s\n\n' % time.ctime())

		# Relation name
		s += '@RELATION rel\n\n'
		
		#print("arff.write header_section attributes")
		# Input attribute specifications
		for fname, ftype in self._features:
			s += '@ATTRIBUTE %-30r %s\n' % (fname, ftype)

		# Label attribute specification
		s += '@ATTRIBUTE %-30r {%s}\n' % ('-label-', ','.join(self._labels))
		#print("arff.write header_section finished")
		return s

	def data_section(self, outfile, tokens, labeled=None):
		"""
		Returns the ARFF data section for the given data.

		:param tokens: a list of featuresets (dicts) or labelled featuresets
			which are tuples (featureset, label).
		:param labeled: Indicates whether the given tokens are labeled
			or not.  If None, then the tokens will be assumed to be
			labeled if the first token's value is a tuple or list.
		"""
		#t1 = time.time()
		#print(0, "arff.write data_section labels")
		
		# Check if the tokens are labeled or unlabeled.  If unlabeled,
		# then use 'None'
		if labeled is None:
			labeled = tokens and isinstance(tokens[0], (tuple, list))
		if not labeled:
			tokens = [(tok, None) for tok in tokens]
		
		#t2 = time.time()
		#print('%.1f' % (t2 - t1), "arff.write data_section data")
		
		# Data section
		outfile.write('\n@DATA\n')
		for (tok, label) in tokens:
			s = ','.join(['%s' % tok.get(fname) for fname, ftype in self._features])
			if label is None:
				s += ',?\n'
			else:
				s += ',%s\n' % label
			outfile.write(s)
		
		#t3 = time.time()
		#print('%.1f' % (t3 - t2), "arff.write data_section finished")
		return

	def _fmt_arff_val(self, fval):
		if fval is None:
			return '?'
		elif isinstance(fval, (bool, compat.integer_types)):
			return '%s' % fval
		elif isinstance(fval, float):
			return '%r' % fval
		else:
			return '%r' % fval


if __name__ == '__main__':
	# from nltk.classify.util import names_demo, binary_names_demo_features
	# def make_classifier(featuresets):
		# return WekaClassifier.train('/tmp/name.model', featuresets,
									# 'C4.5')
	# classifier = names_demo(make_classifier, binary_names_demo_features)
	pass
