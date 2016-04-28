import os
import xml.etree.ElementTree as ET
from lxml import html
from bs4 import BeautifulSoup
from pprint import pprint as pprint
from arktwokenizepy import twokenize
from urllib.parse import urlparse, urlunparse

from author2015 import * # includes the Python classes LabeledAuthor and UnlabeledAuthor

def trim_url(raw_url):
	""" Returns just the scheme and network location part of a url (must begin with a valid scheme like http), for example:
	'http://t.co/v9E09cqCKn' => 't.co'
	'http://4sq.com/5w0x9q' => '4sq.com'
	'4sq.com/5w0x9q' => ''
	'www.google.com' => ''
	"""
	parsed_url = urlparse(raw_url)
	return urlunparse((parsed_url.scheme, parsed_url.netloc, '', '', '', ''))

def my_preprocess(doctext):
	"""Performs preprocessing/cleanup and tokenization of a single tweet, returns a list of tokens"""
	words = []

	doctext = doctext.strip()

	# 2016 update: don't necessarily want to lowercase everything at this stage
	# for word in twokenize.tokenize(doctext.lower()):
	for word in twokenize.tokenize(doctext):

		# 2016 update: not sure if I really want to filter these out or not...
		# if word.startswith("item_") or len(word) < 2:
		# 	# some app seems to post flickr items with item_(sequence of digits)
		# 	continue

		if word.startswith("@"):
			# not sure if it's our responsibility to anonymize usernames if they haven't been already? I guess I will
			word = "@username"

		# Maybe this should be done on non-urls only... not sure
		word = word.replace("‘", "'").replace("’", "'")
		# word = word.replace("…", "...")

		if word.startswith("http"):
			# It's a URL
			word = trim_url(word)
		elif ".com" in word:
			# It's quite likely a URL
			word = trim_url("http://" + word)
		else:
			# Don't treat as a URL
			if "#" in word and not word.startswith("#"):
				print("  '#' found at a position other than the start of a token: {}".format(word))
				print("  tweet was: {}\n".format(doctext))

		words.append(word)

	return words

def get_input_files(inputdir, extension):
	filenames = []
	for root, dirs, files in os.walk(inputdir):
		filenames.extend([os.path.join(root, file) for file in files if file.endswith(extension)])
	print("\nFound", len(filenames), extension, "files in", inputdir)
	return filenames

def language_from_xml_file(filename):
	""" Extracts the language code (one of: 'en', 'es', 'it', 'nl') and author's userid from the specified XML file """
	with open(filename, mode='r', encoding='utf8') as infile:
		soup = BeautifulSoup(infile.read())
		author_id = soup.find('author').get('id')
		author_language = soup.find('author').get('lang')

	if author_id is None: # the PAN 2016 files don't have an id attribute on the author element, so use the base filename without extension
		author_id = os.path.splitext(os.path.basename( filename ))[0] # [0] will get portion of filename before the .xml
	return (author_language, author_id)


def group_by_language(inputfilenames, config_year="2015"):
	print("\ngroup_by_language")
	if config_year == "2015":
		filenames_by_language = {'en': [], 'es': [], 'it': [], 'nl': []}
		userids_by_language = {'en': [], 'es': [], 'it': [], 'nl': []}
	else:
		filenames_by_language = {'en': [], 'es': []}
		userids_by_language = {'en': [], 'es': []}
	
	processed_file_count = 0
	
	for filename in inputfilenames:
		#print(" [File {current} of {total}]\t{filename}".format(current=processed_file_count+1, total=len(inputfilenames), filename=os.path.basename(filename)))
		file_language, userid = language_from_xml_file(filename)
		file_language = file_language.lower()
		filenames_by_language[file_language].append(filename)
		userids_by_language[file_language].append(userid)
		
		processed_file_count += 1
	return (filenames_by_language, userids_by_language)


def remove_tweet_html(tweet, print_debug=False):
	# assume tweet has already undergone tweet.findtext('.') but not necessarily .strip()
	tweet = tweet.strip()

	if print_debug is True:
		print("\n\t{:16}:".format("Original"), tweet)

	try:
		html_tweet = html.fromstring( tweet )
	except TypeError:
		print('\t--- Not HTMLable! Returning original tweet: {}\n'.format(tweet))
		return tweet
	assert type(html_tweet) == html.HtmlElement

	nohtml_tweet = html.tostring( html_tweet, encoding='unicode', method='text', with_tail=True ).strip()

	if print_debug is True:
		print("\t{:16}: <{}>".format("html.fromstring", html_tweet.tag), html_tweet)
		print("\t{:16}:".format("back tostring"), nohtml_tweet)

	return nohtml_tweet


def tweets_from_xml_file(filename, split=""):
	# We don't need to open the file yet; if all goes well, ElementTree will parse the file from just the filename
	# with open(filename, mode='r', encoding='utf8') as infile:
	xml_tree = None
	try:
		xml_tree = ET.ElementTree(file=filename)
	except ET.ParseError as parse_error:
		# ElementTree will try to parse the whole file in one chunk; if that fails due to an invalid character,
		#  we can try to read the file line by line and just blank out the lines (tweets) that have problems.
		# We'll still have the same number of documents to match the <documents> count (as in <documents count="758">),
		#  it just won't yield any tokens when we put it through the tokenizer.
		if not hasattr(parse_error, "position"):
			# If for some reason the ParseError has no two-tuple (line_number, line_offset) called position, just bail.
			print('Encountered unknown ParseError! XMLParser flees!')
			return None
		else:
			error_line_number, error_line_offset = parse_error.position
			print('Encountered ParseError while parsing document: {}'.format(parse_error))

			with open(filename, mode='r', encoding='utf8') as infile:
				lines = infile.readlines()

			bad_line = lines[error_line_number - 1]
			cleaned_line = bad_line[:bad_line.find('<![CDATA')] + '<![CDATA[]]></document>\n'

			print('The offending line ({}) was:'.format(error_line_number - 1))
			print('\t{}'.format(lines[error_line_number - 2].strip()))
			print('old:{}'.format(bad_line.strip()))
			print('new:{}'.format(cleaned_line.strip()))
			print('\t{}'.format(lines[error_line_number - 0].strip()))

			# replace the tweet on line error_line_number-1 (-1 since line numbers start at 1 but lines[] starts at 0) with a blank tweet
			# lines[error_line_number-1] = '<document id="{}" url=""><![CDATA[]]></document>\n'.format(error_line_number)
			lines[error_line_number - 1] = cleaned_line
			print('Replaced tweet on line {} with cleaned tweet: {}'.format(error_line_number, lines[error_line_number-1]))

			old_error_line_number = error_line_number

			while xml_tree is None:
				try:
					xml_tree = ET.fromstringlist(lines)
					print('Successfully parsed the file into an XML tree!\n')
					# Hooray! we can move on!
				except ET.ParseError as parse_error:
					if not hasattr(parse_error, "position"):
						# If for some reason the ParseError has no two-tuple (line_number, line_offset) called position, just bail.
						print('Encountered unknown ParseError! XMLParser flees!')
						return None
					else:
						print(parse_error.position)
						error_line_number, error_line_offset = parse_error.position
						print('Encountered ParseError while parsing document: {}'.format(parse_error))

						bad_line = lines[error_line_number - 1]
						cleaned_line = bad_line[:bad_line.find('<![CDATA')] + '<![CDATA[]]></document>\n'

						print('The offending line ({}) was:'.format(error_line_number - 1))
						print( '\t{}'.format(lines[error_line_number - 2].strip()))
						print('old:{}'.format(bad_line.strip()))
						print('new:{}'.format(cleaned_line.strip()))
						print( '\t{}'.format(lines[error_line_number - 0].strip()))

						if old_error_line_number == error_line_number:
							print('Ack! Going in circles! Flee!')
							return None

						# replace the tweet on line error_line_number-1 (-1 since line numbers start at 1 but lines[] starts at 0) with a blank tweet
						# lines[error_line_number-1] = '<document id="{}" url=""><![CDATA[]]></document>\n'.format(error_line_number)
						lines[error_line_number - 1] = cleaned_line
						print('Replaced tweet on line {} with cleaned tweet: {}'.format(error_line_number, lines[
							error_line_number - 1]))
						old_error_line_number = error_line_number

					# get ready to try forming an xml tree AGAIN

	tweets = [doc.findtext('.').strip()
			  for doc in xml_tree.findall('*document')
			  if len(doc.findtext('.').strip()) > 0]

	print("tweets_from_xml_file: Read", len(tweets), "tweets (using ElementTree)")

	# ONLY FOR INVESTIGATIVE PURPOSES SO WE DON'T GET SWARMED WITH OUTPUT:
	if split == '100' and len(tweets) > 100:
		tweets = tweets[:100]

	"""
	# Begin Investigation time!
	for doc in tweets[:20]:
		thetext = doc.findtext('.')
		print("{:16}:".format("Original"), thetext)

		# fragments = html.fragments_fromstring(thetext)
		# if len(fragments) == 1:
		# 	print("{:16}:".format("Single fragment"), fragments[0])
		# else:
		# 	print("{:16}:".format("Fragments"), fragments)
		# print('--------------------------')

		try:
			thedoc = html.fromstring(thetext)
		except TypeError:
			print('--- Not HTMLable! ---\n')
			continue

		assert type(thedoc) == html.HtmlElement
		# print("{:16}: <{}> ".format("html.fromstring", thedoc.tag), thedoc)

		thedocstring = html.tostring(thedoc, encoding='unicode', method='text', with_tail=True)

		# looks like there can be trailing space here (like &nbsp;), better strip it
		thedocstring = thedocstring.strip()
		print("{:16}:".format("back tostring"), thedocstring)

		print('')

	print('')
	# End of Investigation time!
	"""

	PRINTCOUNT = 0
	if len(tweets) > PRINTCOUNT:
		# don't print full debug info for every single tweet, just the first PRINTCOUNT
		tweets = [ remove_tweet_html(tweet, print_debug=True) for tweet in tweets[:PRINTCOUNT] ] + [ remove_tweet_html(tweet) for tweet in tweets[PRINTCOUNT:] ]
	else:
		tweets = [remove_tweet_html(tweet, print_debug=True) for tweet in tweets]

	if split == 'test':
		return tweets[int(len(tweets)/3)*2:]
	elif  split == 'train':
		return tweets[:int(len(tweets)/3)*2]
	elif split == '100':
		return tweets[:100]
	else:
		return tweets

def write_preprocessed_tweets(language, inputfilenames, tempdir, split=""):
	print("\nwrite_preprocessed_tweets:", language)
	
	processed_file_count = 0
	
	for filename in inputfilenames:
		print("[File {current} of {total}]\t{filename}".format(current=processed_file_count+1, total=len(inputfilenames), filename=os.path.basename(filename)))
		
		tweets = tweets_from_xml_file(filename, split)
		
		if tweets is not None and len(tweets)>0:
			if not isinstance(tweets, list): # if we only read one tweet from a file, make sure we have a list containing the tweet
				tweets = [tweets,]

			outfilename = os.path.join(tempdir, language + "_" + os.path.basename(filename).replace("xml","txt"))
			
			with open(outfilename, 'w', encoding='utf8') as out_file_contents:
				for tweet in tweets:
					token_list = my_preprocess(tweet)

					if len(token_list) > 0:
						newtweet = " ".join(token_list)
						out_file_contents.write(newtweet + "\n")
					else:
						print("Skipping tweet that yielded 0 tokens")
		else:
			print("No valid tweets were found in {filename}".format(filename=filename))
		processed_file_count += 1
		print('--------------------------------------------------\n')
	return


def load_preprocessed_tweets(tweetfilename, force_lowercase=False):
	if not os.path.exists(tweetfilename):
		return []
	
	with open(tweetfilename, 'r', encoding='utf8') as file_contents:
		tweetlines = file_contents.readlines()
	if force_lowercase is True:
		return [tweetline.strip().lower() for tweetline in tweetlines]
	else:
		return [tweetline.strip() for tweetline in tweetlines]


def get_unlabeled_authors(userids_by_language, tempdir, force_lowercase=False, config_year="2015"):
	if config_year == "2015":
		unlabeled_authors = {'en': {}, 'es': {}, 'it': {}, 'nl': {}}
	else:
		unlabeled_authors = {'en': {}, 'es': {}}
	# Holds the information we know about unlabeled authors in the "testing" set, and later our predictions of their attributes.
	# The unlabeled_authors dictionary gets returned to setup_testing.
	print("\nget_unlabeled_authors")
	
	for language in userids_by_language:
		for userid in userids_by_language[language]:
			tweetfilename = os.path.join(tempdir, language + "_" + userid + ".txt")
			tweets = load_preprocessed_tweets(tweetfilename, force_lowercase)

			if len(tweets) > 0:
				unlabeled_authors[language][userid] = UnlabeledAuthor(language, tweets, userid)
				# print(unlabeled_authors[language][userid])
			else:
				print("No tweets remained after preprocessing for author:", userid)
			
	return unlabeled_authors


def get_labeled_authors(truthfilenames, userids_by_language, tempdir, force_lowercase=False, config_year="2015"):
	if config_year == "2015":
		labeled_authors = {'en': {}, 'es': {}, 'it': {}, 'nl': {}}
	else:
		labeled_authors = {'en': {}, 'es': {}}
	# Holds the information we know about labeled authors in the "training" set.
	# The labeled_authors dictionary gets returned to setup_training.
	
	for filename in truthfilenames:
		print("\nget_labeled_authors truth file:", filename)
		print("get_labeled_authors config_year:", config_year)
		for language in userids_by_language:
			print('  {count} {lang} usernames'.format(count=len(userids_by_language['en']), lang=language))

		with open(filename, 'r', encoding='utf8') as truth_contents:
			''' example truth.txt line:
			user679:::F:::18-24:::0.0:::-0.2:::0.1:::0.0:::0.3
			{userid}:::{gender}:::{age_group}:::{extroverted}:::{stable}:::{agreeable}:::{conscientious}:::{open}
			'''
			rows = truth_contents.readlines()
			
			for row in rows:
				row = row.strip()
				row_data = row.split(':::')
				
				# the 2014 training data used gender labels "MALE" and "FEMALE" instead of "M" and "F"; change to use "M" and "F"
				if row_data[1] == "MALE":
					row_data[1] = "M"
				elif row_data[1] == "FEMALE":
					row_data[1] = "F"

				if config_year == "2014":
					if row_data[2] == "50-64" or row_data[2] == "65-xx": # group 2014 author data into the same available age classes in 2015
						row_data[2] = "50-XX"
					# 2014 data won't have the five personality traits, so use None for the next 5 values
					row_data.extend([None, None, None, None, None])
				
				# the item at index 2 is age_group, which is always "XX-XX" for dutch and italian. instead of "XX-XX", use None
				if config_year == "2015" and row_data[2] == "XX-XX":
					row_data[2] = None

				# row_data[0] is the author's userid
				userid = row_data[0]
				
				# find which language an author belongs to
				for language in userids_by_language:
					if userid in userids_by_language[language]:
						# if this userid is in the list of authors labeled with this language...
						
						tweetfilename = os.path.join(tempdir, language + "_" + userid + ".txt")
						tweets = load_preprocessed_tweets(tweetfilename, force_lowercase)

						if len(tweets) > 0:
							labeled_authors[language][userid] = LabeledAuthor(language, tweets, *row_data)
							print(labeled_authors[language][userid])
						else:
							print("(get_labeled_authors) No tweets remained after preprocessing for author:", userid)
						
						break # stop the "for language in userids_by_language" loop
				
	return labeled_authors