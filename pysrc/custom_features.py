from numpy import zeros, size
from arktopics import word_topic_dict, max_topic_index
from top100_personality_phrases import *
from vaderSentiment.vaderSentiment import sentiment


def sentiment_positive(tweet_text, language):
	if language is "en":
		return sentiment(tweet_text)['pos']
	else:
		return float(0)

def sentiment_negative(tweet_text, language):
	if language is "en":
		return sentiment(tweet_text)['neg']
	else:
		return float(0)

def sentiment_neutral(tweet_text, language):
	if language is "en":
		return sentiment(tweet_text)['neu']
	else:
		return float(0)

def sentiment_compound(tweet_text, language):
	if language is "en":
		return sentiment(tweet_text)['compound']
	else:
		return float(0)

def word_count(tweet_text, language):
	return len(tweet_text.split(" "))

def word_length_avg(tweet_text, language):
	tweet_words = tweet_text.split(" ")
	word_length = sum( [len(word) for word in tweet_words] ) / len(tweet_words)
	return word_length

def word_length_max(tweet_text, language):
	tweet_words = tweet_text.split(" ")
	word_length = max( [len(word) for word in tweet_words] )
	return word_length

def words_with_punctuation(tweet_text, language):
	tweet_words = tweet_text.split(" ")
	return len([word for word in tweet_words if not word.isalpha()]) / len(tweet_words)


def punctuation_chars(tweet_text, language):
	punctuation_chars = 0
	nonspace_chars = 0
	for char in tweet_text:
		if not char.isspace():
			nonspace_chars += 1
			if not char.isalnum():
				punctuation_chars += 1
	return punctuation_chars / nonspace_chars

def digit_chars(tweet_text, language):
	digit_chars = 0
	nonspace_chars = 0
	for char in tweet_text:
		if not char.isspace():
			nonspace_chars += 1
			if char.isdigit():
				digit_chars += 1
	return digit_chars / nonspace_chars

def accent_chars(tweet_text, language):
	"""
	Recognizes characters coded 192 to 255 in common western european character encodings (such as Windows-1252 or ISO/IEC 8859-1).
	This includes things like: Á, Ñ, ç, ß, ä
	"""
	accent_chars = 0
	nonspace_chars = 0
	for char in tweet_text:
		if not char.isspace():
			nonspace_chars += 1
			if ord(char) >= 192 and ord(char) <= 255:
				accent_chars += 1
	return accent_chars / nonspace_chars

def hashtag_count(tweet_text, language):
	tweet_words = tweet_text.split(" ")
	return len( [word for word in tweet_words if word.startswith("#")] ) / len(tweet_words)

def username_count(tweet_text, language):
	tweet_words = tweet_text.split(" ")
	return len( [word for word in tweet_words if word.startswith("@")] ) / len(tweet_words)

def url_count(tweet_text, language):
	tweet_words = tweet_text.split(" ")
	return len( [word for word in tweet_words if word.startswith("http") or word.startswith("www")] ) / len(tweet_words)

def firstword_username(tweet_text, language):
	return int(tweet_text[0] == "@")

def firstword_hashtag(tweet_text, language):
	return int(tweet_text[0] == "#")

def firstword_url(tweet_text, language):
	return int(tweet_text.startswith("http") or tweet_text.startswith("www"))

def lastword_username(tweet_text, language):
	lastword = tweet_text.split(" ")[-1] # if tweet_text is only one word, this will simply be that word
	return int(lastword[0] == "@")

def lastword_hashtag(tweet_text, language):
	lastword = tweet_text.split(" ")[-1] # if tweet_text is only one word, this will simply be that word
	return int(lastword[0] == "#")

def lastword_url(tweet_text, language):
	lastword = tweet_text.split(" ")[-1] # if tweet_text is only one word, this will simply be that word
	return int(lastword.startswith("http") or lastword.startswith("www"))

# ---------------------------------------------------------------------
def low_extroverted_phrases(tweet_text, language):
	tweet_word_count = len(tweet_text.split(" "))
	tweet_text = " " + tweet_text.lower() + " "
	personality_match = 0
	for phrase, phrase_weight in personality_phrases[language]['extroverted']['low_phrases'].items():
		if phrase in tweet_text:
			personality_match += phrase_weight
	personality_match /= tweet_word_count
	personality_match /= personality_phrases[language]['extroverted']['low_maxweight']
	return personality_match

def high_extroverted_phrases(tweet_text, language):
	tweet_word_count = len(tweet_text.split(" "))
	tweet_text = " " + tweet_text.lower() + " "
	personality_match = 0
	for phrase, phrase_weight in personality_phrases[language]['extroverted']['high_phrases'].items():
		if phrase in tweet_text:
			personality_match += phrase_weight
	personality_match /= tweet_word_count
	personality_match /= personality_phrases[language]['extroverted']['high_maxweight']
	return personality_match


def low_stable_phrases(tweet_text, language):
	tweet_word_count = len(tweet_text.split(" "))
	tweet_text = " " + tweet_text.lower() + " "
	personality_match = 0
	for phrase, phrase_weight in personality_phrases[language]['stable']['low_phrases'].items():
		if phrase in tweet_text:
			personality_match += phrase_weight
	personality_match /= tweet_word_count
	personality_match /= personality_phrases[language]['stable']['low_maxweight']
	return personality_match

def high_stable_phrases(tweet_text, language):
	tweet_word_count = len(tweet_text.split(" "))
	tweet_text = " " + tweet_text.lower() + " "
	personality_match = 0
	for phrase, phrase_weight in personality_phrases[language]['stable']['high_phrases'].items():
		if phrase in tweet_text:
			personality_match += phrase_weight
	personality_match /= tweet_word_count
	personality_match /= personality_phrases[language]['stable']['high_maxweight']
	return personality_match


def low_agreeable_phrases(tweet_text, language):
	tweet_word_count = len(tweet_text.split(" "))
	tweet_text = " " + tweet_text.lower() + " "
	personality_match = 0
	for phrase, phrase_weight in personality_phrases[language]['agreeable']['low_phrases'].items():
		if phrase in tweet_text:
			personality_match += phrase_weight
	personality_match /= tweet_word_count
	personality_match /= personality_phrases[language]['agreeable']['low_maxweight']
	return personality_match

def high_agreeable_phrases(tweet_text, language):
	tweet_word_count = len(tweet_text.split(" "))
	tweet_text = " " + tweet_text.lower() + " "
	personality_match = 0
	for phrase, phrase_weight in personality_phrases[language]['agreeable']['high_phrases'].items():
		if phrase in tweet_text:
			personality_match += phrase_weight
	personality_match /= tweet_word_count
	personality_match /= personality_phrases[language]['agreeable']['high_maxweight']
	return personality_match


def low_conscientious_phrases(tweet_text, language):
	tweet_word_count = len(tweet_text.split(" "))
	tweet_text = " " + tweet_text.lower() + " "
	personality_match = 0
	for phrase, phrase_weight in personality_phrases[language]['conscientious']['low_phrases'].items():
		if phrase in tweet_text:
			personality_match += phrase_weight
	personality_match /= tweet_word_count
	personality_match /= personality_phrases[language]['conscientious']['low_maxweight']
	return personality_match

def high_conscientious_phrases(tweet_text, language):
	tweet_word_count = len(tweet_text.split(" "))
	tweet_text = " " + tweet_text.lower() + " "
	personality_match = 0
	for phrase, phrase_weight in personality_phrases[language]['conscientious']['high_phrases'].items():
		if phrase in tweet_text:
			personality_match += phrase_weight
	personality_match /= tweet_word_count
	personality_match /= personality_phrases[language]['conscientious']['high_maxweight']
	return personality_match


def low_open_phrases(tweet_text, language):
	tweet_word_count = len(tweet_text.split(" "))
	tweet_text = " " + tweet_text.lower() + " "
	personality_match = 0
	for phrase, phrase_weight in personality_phrases[language]['open']['low_phrases'].items():
		if phrase in tweet_text:
			personality_match += phrase_weight
	personality_match /= tweet_word_count
	personality_match /= personality_phrases[language]['open']['low_maxweight']
	return personality_match

def high_open_phrases(tweet_text, language):
	tweet_word_count = len(tweet_text.split(" "))
	tweet_text = " " + tweet_text.lower() + " "
	personality_match = 0
	for phrase, phrase_weight in personality_phrases[language]['open']['high_phrases'].items():
		if phrase in tweet_text:
			personality_match += phrase_weight
	personality_match /= tweet_word_count
	personality_match /= personality_phrases[language]['open']['high_maxweight']
	return personality_match

# ---------------------------------------------------------------------

def female_phrases(tweet_text, language):
	tweet_word_count = len(tweet_text.split(" "))
	tweet_text = " " + tweet_text.lower() + " "
	personality_match = 0
	for phrase, phrase_weight in gender_phrases[language]['female']['phrases'].items():
		if phrase in tweet_text:
			personality_match += phrase_weight
	personality_match /= tweet_word_count
	personality_match /= gender_phrases[language]['female']['maxweight']
	return personality_match

def male_phrases(tweet_text, language):
	tweet_word_count = len(tweet_text.split(" "))
	tweet_text = " " + tweet_text.lower() + " "
	personality_match = 0
	for phrase, phrase_weight in gender_phrases[language]['male']['phrases'].items():
		if phrase in tweet_text:
			personality_match += phrase_weight
	personality_match /= tweet_word_count
	personality_match /= gender_phrases[language]['male']['maxweight']
	return personality_match

# ---------------------------------------------------------------------

all_feature_prefixes = ['topic', 'word', 'percent', 'is', 'low', 'high', 'gender', 'sentiment']

custom_feature_defs = (
	("word_count", "NUMERIC", word_count ),
	("word_length_avg", "NUMERIC", word_length_avg ),
	("word_length_max", "NUMERIC", word_length_max ),

	("percent_punctuated_words", "NUMERIC", words_with_punctuation ),
	("percent_url_words", "NUMERIC", url_count ),
	("percent_hashtag_words", "NUMERIC", hashtag_count),
	("percent_username_words", "NUMERIC", username_count),

	("percent_punctuation_chars", "NUMERIC", punctuation_chars ),
	("percent_accent_chars", "NUMERIC", accent_chars ),
	("percent_digit_chars", "NUMERIC", digit_chars ),

	("is_firstword_username", "INTEGER", firstword_username),
	("is_firstword_hashtag", "INTEGER", firstword_hashtag),
	("is_firstword_url", "INTEGER", firstword_url),
	("is_lastword_username", "INTEGER", lastword_username),
	("is_lastword_hashtag", "INTEGER", lastword_hashtag),
	("is_lastword_url", "INTEGER", lastword_url),

	("low_extroverted_phrases", "NUMERIC", low_extroverted_phrases),
	("high_extroverted_phrases", "NUMERIC", high_extroverted_phrases),

	("low_stable_phrases", "NUMERIC", low_stable_phrases),
	("high_stable_phrases", "NUMERIC", high_stable_phrases),

	("low_agreeable_phrases", "NUMERIC", low_agreeable_phrases),
	("high_agreeable_phrases", "NUMERIC", high_agreeable_phrases),

	("low_conscientious_phrases", "NUMERIC", low_conscientious_phrases),
	("high_conscientious_phrases", "NUMERIC", high_conscientious_phrases),

	("low_open_phrases", "NUMERIC", low_open_phrases),
	("high_open_phrases", "NUMERIC", high_open_phrases),

	("gender_female_phrases", "NUMERIC", female_phrases),
	("gender_male_phrases", "NUMERIC", male_phrases),

	("sentiment_negative", "NUMERIC", sentiment_negative),
	("sentiment_positive", "NUMERIC", sentiment_positive),
	("sentiment_neutral", "NUMERIC", sentiment_neutral),
	("sentiment_compound", "NUMERIC", sentiment_compound),
)
custom_feature_names = [custom_feature_name for custom_feature_name, _, _ in custom_feature_defs]

def custom_feature_by_name(name):
	for feature_triple in custom_feature_defs:
		custom_feature_name, custom_feature_type, custom_feature_function = feature_triple
		if custom_feature_name == name:
			return feature_triple
	raise KeyError # if we didn't find an entry in custom_feature_defs with the specified name

def extract_instance_features(authors, feature_names, combinetweets=True):
	if combinetweets:
		featurized_instances = get_fbtopic_dists_joinedtweets(authors, feature_names)
	else:
		featurized_instances = get_fbtopic_dists(authors, feature_names)
	return featurized_instances

def get_weka_features(authors, feature_names):
	weka_features = []
	if authors is not None:
		weka_features.insert(0,('author_id', [author_id for author_id in authors]))

	if len(feature_names) > 0:
		for name in feature_names:
			# the liac-arff ArffEncoder is going to put quotes around attributes that contain any of these characters,
			# so don't put our own quotes around them first or we'll end up with feature names like ""tw_100%""
			if any([nope in name for nope in (' ', '%', '{', '}', ',')]):
				# print("AVOIDING DOUBLE DOUBLE-QUOTES", name, "becomes", name.strip('"'))
				name = name.strip('"')
			if name in custom_feature_names:
				custom_feature = custom_feature_by_name(name)
				weka_features.append((name, custom_feature[1])) # the second element of the triple is the feature type
			else:
				weka_features.append((name, 'NUMERIC'))
	return weka_features

def get_fbtopic_dists(authors, feature_names):
	"""
		tweet_feature_vector has as many elements as there are topics (2000) **plus number of "custom" features in custom_feature_defs**
		each topic element should be the conditional probability of that topic given this document (tweet).
		for now, just loot the cond. probabilities out of the (term,category,weight) csv, but only for
			terms significant enough to make the cutoff into the "top 20 terms per topic" csv.
		since tweets are so short anyways, we'll say that the term frequency within this document is 1 if it occurs at all.
		then the contribution of each word in this tweet (if it exists in the table at all) is 1 * p(topic|term) * 1/doclength,
			so just sum the conditional probabilities for whatever terms exist in our table, and multiply whole thing by 1/doclength
	"""
	# word_topic_dict = dict(word_topic_map)
	# word_topic_dict['dictionary'] -> (535, 1019)

	author_topic_dists = {}

	for author_id in sorted(authors):
		author_data = authors[author_id]

		nrows = len(author_data.tweets)
		ncols = len(feature_names)
		tweet_feature_vectors = zeros([nrows, ncols], dtype=float)

		#author_nonzero_elements = [] # keep track of how many dimensions of the tweet feature vector are non-zero, for tuning purposes

		for row_index, tweet in enumerate(author_data.tweets):
			tweet_words = tweet.lower().split(" ")
			# tweet_feature_vector = zeros([len(feature_names)], dtype=float)

			if any([feature.startswith('topic') for feature in feature_names]):
				for word in tweet_words:
					if word in word_topic_dict:
						if isinstance(word_topic_dict[word], list):
							for topic_number in word_topic_dict[word]:
								tweet_feature_vectors[row_index, topic_number] += 1 #condprob_dict[(word, topic_number)]
						else:
							tweet_feature_vectors[row_index, word_topic_dict[word]] += 1

			# the elements of tweet_feature_vector that correspond to non-topic features are still zero,
			# so this is fine to do over the whole vector
			###tweet_feature_vectors[row_index, :] /= len(tweet_words)

			# now that we're done messing with the topic features, assign values for other features
			for feature_name in [feature for feature in feature_names if not feature.startswith('topic')]:
				feature_vector_index = feature_names.index(feature_name)
				if feature_name in custom_feature_names:
					_, _, custom_feature_function = custom_feature_by_name(feature_name)
					tweet_feature_vectors[row_index, feature_vector_index] = custom_feature_function(tweet, author_data.language)

			#tweet_feature_vectors.append(tweet_feature_vector)
			#author_nonzero_elements.append(size(tweet_feature_vector.nonzero()))

		author_topic_dists.update({author_id: tweet_feature_vectors})
		# print(author_id, nrows * ncols, "entries in feature matrix, % nonzero:", round(size(tweet_feature_vectors.nonzero())/(nrows * ncols),2)  )
	return author_topic_dists


def get_fbtopic_dists_joinedtweets(authors, feature_names):
	# word_topic_dict = dict(word_topic_map)
	# word_topic_dict['dictionary'] -> (535, 1019)

	author_topic_dists = {}

	for author_id in authors:
		author_data = authors[author_id]

		author_nonzero_elements = []
		tweet_feature_vector = zeros([len(feature_names)], dtype=float) # we'll use the last len(custom_feature_defs) elements for custom features
		for tweet in author_data.tweets:
			tweet_words = tweet.lower().split(" ")

			"""
			if any([feature.startswith('topic') for feature in feature_names]):
				for word in tweet_words:
					if word in word_topic_dict:
						for topic_number in word_topic_dict[word]:
							tweet_feature_vectors[row_index, topic_number] += 1 #condprob_dict[(word, topic_number)]
			"""

			if any([feature.startswith('topic') for feature in feature_names]):
				for word in tweet_words:
					if word in word_topic_dict:
						if isinstance(word_topic_dict[word], list):
							for topic_number in word_topic_dict[word]:
								tweet_feature_vector[topic_number] += 1 #condprob_dict[(word, topic_number)] / len(tweet_words)
						else:
							tweet_feature_vector[word_topic_dict[word]] += 1

			# now that we're done messing with the topic features, assign values for other features
			for feature_name in [feature for feature in feature_names if not feature.startswith('topic')]:
				feature_vector_index = feature_names.index(feature_name)
				if feature_name in custom_feature_names:
					_, _, custom_feature_function = custom_feature_by_name(feature_name)
					tweet_feature_vector[feature_vector_index] = custom_feature_function(tweet, author_data.language)

			author_nonzero_elements.append(size(tweet_feature_vector.nonzero()))

		# Only normalize by number of author tweets in the "combinetweets" version of this function
		tweet_feature_vector /= len(author_data.tweets)

		author_topic_dists.update({author_id: [tweet_feature_vector,]})
		# print(author_id, len(author_data.tweets), "tweets,\taverage nonzero features per instance:", round(sum(author_nonzero_elements)/len(author_nonzero_elements), 2))
	return author_topic_dists