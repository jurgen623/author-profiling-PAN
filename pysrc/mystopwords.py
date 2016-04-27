import os

# Top ~20 words from lists on http://en.wiktionary.org/wiki/Wiktionary:Frequency_lists
wiktionary_stopwords = {
	'en': ['you', 'ya', 'i', 'to', 'the', 'a', 'and', 'that', 'it', 'of', 'me', 'what', 'is', 'in', 'this', 'know', "i'm", 'for', 'no', 'have', 'my', "don't", 'like'],
	'es': ['que', 'de', 'no', 'a', 'la', 'el', 'es', 'y', 'en', 'lo', 'un', 'por', 'qué', 'me', 'una', 'te', 'los', 'se', 'con', 'para', 'mi', 'está'],
	'it': ['non', 'di', 'che', 'è', 'e', 'la', 'il', 'un', 'a', 'per', 'in', 'una', 'mi', 'sono', 'ho', 'ma', "l'", 'lo', 'ha', 'le', 'si', 'ti', 'con', 'cosa'],
	'nl': ['ik', 'je', 'het', "'t", 'de', 'dat', 'is', 'een', 'niet', 'en', 'wat', 'van', 'we', 'in', 'ze', 'hij', 'op', 'te', 'zijn', 'er', 'maar']
}

empty_tokens = ['', "''", '""']

# Read in stopwords list from the NLTK "Stopwords" corpus
stopwords_dir = os.path.join('include', 'stopwords')

my_stopwords = {'en': [], 'es': [], 'it': [], 'nl': []}

for language in my_stopwords.keys():
	with open(os.path.join(stopwords_dir, language + ".txt"), mode='r', encoding='utf8') as stopwords_file:
		lines = stopwords_file.readlines()
		my_stopwords[language] = list(set( [line.strip() for line in lines] + wiktionary_stopwords[language] + empty_tokens ))
	# print(my_stopwords[language])