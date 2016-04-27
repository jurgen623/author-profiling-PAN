from config2015 import *
from author_truths import author_truth_dict
import arff


prediction_file_template = \
"""<author id="{userid}"
	type="twitter"
	lang="{language}"
	age_group="{age_group}"
	gender="{gender}"
	extroverted="{extroverted:0.4}"
	stable="{stable:0.4}"
	agreeable="{agreeable:0.4}"
	conscientious="{conscientious:0.4}"
	open="{open:0.4}"
	/>
"""

prediction_file_template_2014 = \
"""<author id="{userid}"
	type="twitter"
	lang="{language}"
	age_group="{age_group}"
	gender="{gender}"
	/>
"""


class ArffDict(dict):
	""" For use with the liac-arff python library (imported as 'arff')
	xor_dataset = {
		'description': 'XOR Dataset',
		'relation': 'XOR',
		'attributes': [
			('input1', 'REAL'),
			('input2', 'REAL'),
			('y', 'REAL'),
		],
		'data': [
			[0.0, 0.0, 0.0],
			[0.0, 1.0, 1.0],
			[1.0, 0.0, 1.0],
			[1.0, 1.0, 0.0]
		]
	}
	"""
	def __init__(self, relation, attributes, data, description="", **kwargs):
		"""
			relation: (OBLIGATORY) a string with the name of the dataset
			attributes: (OBLIGATORY) a list of attributes with the following template:
					(attribute_name, attribute_type)
				the attribute_name is a string, and attribute_type must be an string or a list of strings
			data: (OBLIGATORY) a list of data instances. Each data instance must be a list with values,
				depending on the attributes
			description: (OPTIONAL) a string with the description of the dataset
			"""
		super().__init__(**kwargs)

		if not isinstance(relation, str):
			relation = repr(relation)
		if len(attributes) < 1:
			print("len(attributes) < 1")
			raise ValueError
		elif len(data) < 1:
			print("len(data) < 1")
			raise ValueError
		elif not all([isinstance(attribute, (list, tuple)) for attribute in attributes]):
			print("Not all attributes were lists or tuples. Some were of types", set(map(type, data)))
			raise ValueError
		elif not all([len(row) == len(attributes) for row in data]):
			print("Not all rows have the same number of elements as there are attributes ({0}). Some have".format(len(attributes)), set(map(len, data)), "attributes")
			print("First attributes:", attributes[:3])
			print("Last attributes:", attributes[-3:])
			# print([row for row in data if len(row) != len(attributes)])
			raise ValueError
		self['relation'] = relation
		self['attributes'] = attributes
		self['data'] = data
		self['description'] = description

class MyArffEncoder(arff.ArffEncoder):
	def _encode_data(self, data):
		'''(INTERNAL) Encodes a line of data.

		Data instances encoded in sparse arff format. For example, if there
		are 2021 attributes and for some instance, all the values are 0 except for those
		of the attributes at indices 0, 668, 2007, 2013, 2014, and 2020, the line is:
			{0 user1,668 0.068302779,2007 1,2013 0.5,2014 1,2020 M}

		:param data: a list of values.
		:return: a string with the encoded data line.
		'''
		new_data = []
		for attribute_index, value in enumerate(data):
			if value is not None and value == 0:
				continue

			if value is None or value == u'':
				value = "?"
			elif isinstance(value, str):
				for escape_char in arff._ESCAPE_DCT:
					if escape_char in value:
						value = arff.encode_string(value)
						break
			else:
				value = round(value,10)

			s = "{ind} {val}".format(ind=attribute_index, val=value)
			new_data.append(s)

		return "{{{0}}}".format(u','.join(new_data))

class UnlabeledAuthor(object):
	""" Holds a list of tweets by a testing set author (to start, we only know their userid and language)
		Once we make our predictions as to the rest of their attributes, we will store those here also """
	def __init__(self, language, tweets, userid):
		if language not in LANGUAGES:
			print("Invalid language: " + repr(language))
			return

		self.language = language
		self.tweets = tweets
		self.userid = userid
		
		self.gender = None
		self.age_group = None
		self.personality = dict.fromkeys(PERSONALITY_TRAITS) # the values will be None by default

		self.vote_counts = {'gender': {'M': 0, 'F': 0}, 'age_group': {'18-24': 0, '25-34': 0, '35-49': 0, '50-XX': 0}}
		
	def __repr__(self):
		personality_values = []
		for trait_value in self.traits:
			if trait_value is None:
				personality_values.append("?")
			else:
				personality_values.append( int(trait_value*10) )
		personality_format = "({0:2},{1:2},{2:2},{3:2},{4:2})"
		
		author_format = "<UnlabeledAuthor: {lang} {userid:8} {gender}, {age}, {person}, {twcount:3} tweets>"
		
		if self.gender is None:
			repr_gender = "?"
		else:
			repr_gender = self.gender
			
		if self.age_group is None:
			repr_age_group = "?"
		else:
			repr_age_group = self.age_group
			
		return author_format.format(lang=self.language, userid=self.userid, gender=repr_gender, age=repr_age_group,
								  person=personality_format.format(*personality_values), twcount=len(self.tweets))


	@property
	def traits(self):
		return tuple([self.personality[trait] for trait in PERSONALITY_TRAITS])
		
	def make_prediction(self, attribute_name, attribute_value):
		if attribute_name == "gender":
			if self.gender is not None:
				if self.gender != attribute_value:
					print("Old: {prev}\t New: {new}".format(prev=self.gender, new=attribute_value))
			self.gender = attribute_value
		elif attribute_name == "age_group":
			if self.age_group is not None:
				if self.age_group != attribute_value:
					print("Old: {prev}\t New: {new}".format(prev=self.age_group, new=attribute_value))
			self.age_group = attribute_value
		elif attribute_name in PERSONALITY_TRAITS:
			if self.personality[attribute_name] is not None:
				attribute_value = (attribute_value + self.personality[attribute_name])/2
				if abs(self.personality[attribute_name] - attribute_value) > 0.05:
					print(" Old: {prev:4}\t New: {new:4}".format(prev=self.personality[attribute_name], new=attribute_value))
			self.personality[attribute_name] = attribute_value
			
	def get_prediction_xml(self):
		if self.age_group is None:
			age_group = 'XX-XX'
		else:
			age_group = self.age_group

		# the TIRA evaluator tool doesn't recognize "M" as "male" and "F" as "female", so just write the full word
		if self.gender == "M":
			unabbreviated_gender = "male"
		elif self.gender == "F":
			unabbreviated_gender = "female"
		else:
			unabbreviated_gender = "?"

		if None in self.traits: # must be a 2014 author reformatted for 2015
			prediction_file_contents = prediction_file_template_2014.format(
				userid = self.userid,
				language = self.language,
				age_group = age_group,
				gender = unabbreviated_gender
			)
		else:
			prediction_file_contents = prediction_file_template.format(
				userid = self.userid,
				language = self.language,
				age_group = age_group,
				gender = unabbreviated_gender,
				extroverted = self.personality['extroverted'],
				stable = self.personality['stable'],
				agreeable = self.personality['agreeable'],
				conscientious = self.personality['conscientious'],
				open = self.personality['open']
			)
		return prediction_file_contents

class LabeledAuthor(object):
	""" Holds a list of tweets by a training set author (to start, we know their userid, language, and all attribute values) """
	def __init__(self, language, tweets, userid, gender, age_group, ext, sta, agr, con, opn):
		if language not in LANGUAGES:
			print("Invalid language: " + repr(language))
			return
		if gender not in GENDERS:
			print("Invalid gender: " + repr(gender))
			return
		if age_group not in AGE_GROUPS + (None,):
			print("Invalid age_group: " + repr(age_group))
			return

		self.language = language
		self.tweets = tweets
		self.userid = userid
		self.gender = gender
		self.age_group = age_group
		self.personality = dict.fromkeys(PERSONALITY_TRAITS)

		for trait_name, trait_value in zip(PERSONALITY_TRAITS, (ext, sta, agr, con, opn)):
			if trait_value is None: # allowed for 2014 author data reformatted for 2015
				self.personality[trait_name] = None
			else:
				try:
					float_trait = float(trait_value)
					self.personality[trait_name] = float_trait
				except ValueError:
					print('Could not convert {name} to float: {value}'.format(name=trait_name, value=trait_value))
					return
	
	def __repr__(self):
		if None in self.traits:  # must be a 2014 author reformatted for 2015
			personality_str = "(no personality data from 2014)"
		else:
			personality_values = [int(trait*10) for trait in self.traits]
			personality_format = "({0:2},{1:2},{2:2},{3:2},{4:2})"
			personality_str = personality_format.format(*personality_values)
		author_format = "<LabeledAuthor: {lang} {userid:8} {gender}, {age}, {person}, {twcount:3} tweets>"
		return author_format.format(lang=self.language, userid=self.userid, gender=self.gender, age=self.age_group,
								  person=personality_str, twcount=len(self.tweets))
	
	def get_label(self, class_attribute):
		if class_attribute == "gender":
			return self.gender
		elif class_attribute == "age_group" and self.age_group is not None:
			return self.age_group
		elif class_attribute == "age_group" and self.age_group is None:
			return "XX-XX"
		elif class_attribute in PERSONALITY_TRAITS:
			if self.personality[class_attribute] is None: # must be a 2014 author reformatted for 2015
				return None
			else:
				return '%.1f' % self.personality[class_attribute]
	
	@property
	def traits(self):
		return tuple([self.personality[trait] for trait in PERSONALITY_TRAITS])

	def get_truth_dict(self):
		"""
		Example: We want to form the dictionary of author_truth_dict['user1'] below
		author_truth_dict = {
			'user1': {'age_group': '50-XX',
					  'agreeable': '0.5',
					  'conscientious': '0.2',
					  'extroverted': '0.4',
					  'gender': 'M',
					  'language': 'en',
					  'open': '0.3',
					  'stable': '0.3',
					  'userid': 'user1'},
					  ....
		"""
		if self.age_group is None:
			age_group = 'XX-XX'
		else:
			age_group = self.age_group
		truth_dict = {'age_group': age_group,
					  'agreeable': self.personality['agreeable'],
					  'conscientious': self.personality['conscientious'],
					  'extroverted': self.personality['extroverted'],
					  'gender': self.gender,
					  'language': self.language,
					  'open': self.personality['open'],
					  'stable': self.personality['stable'],
					  'userid': self.userid}
		return truth_dict
	
	def get_prediction_xml(self):
		if self.age_group is None:
			age_group = 'XX-XX'
		else:
			age_group = self.age_group
		
		# the TIRA evaluator tool doesn't recognize "M" as "male" and "F" as "female", so just write the full word
		if self.gender == "M":
			unabbreviated_gender = "male"
		elif self.gender == "F":
			unabbreviated_gender = "female"
		else:
			unabbreviated_gender = "?"

		if None in self.traits: # must be a 2014 author reformatted for 2015
			prediction_file_contents = prediction_file_template_2014.format(
				userid = self.userid,
				language = self.language,
				age_group = age_group,
				gender = unabbreviated_gender
			)
		else:
			prediction_file_contents = prediction_file_template.format(
				userid = self.userid,
				language = self.language,
				age_group = age_group,
				gender = unabbreviated_gender,
				extroverted = self.personality['extroverted'],
				stable = self.personality['stable'],
				agreeable = self.personality['agreeable'],
				conscientious = self.personality['conscientious'],
				open = self.personality['open']
			)
		return prediction_file_contents
