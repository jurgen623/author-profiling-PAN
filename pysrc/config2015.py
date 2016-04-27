CONFIG_YEAR = "2015"

LANGUAGES = ('en', 'es', 'it', 'nl')
GENDERS = ('M', 'F')
AGE_GROUPS = ('18-24', '25-34', '35-49', '50-XX')
PERSONALITY_TRAITS = ('extroverted', 'stable', 'agreeable', 'conscientious', 'open')

labels_for_attribute = {
	'gender': GENDERS,
	'age_group': AGE_GROUPS,
	'extroverted': 'NUMERIC',
	'stable': 'NUMERIC',
	'agreeable': 'NUMERIC',
	'conscientious': 'NUMERIC',
	'open': 'NUMERIC',
	}

#run_languages = ('en', 'es') # edit this value to control which languages to run during development
# In development: try class labels of <gender>_<age>_personality_trait_index<trait[personality_trait_index]>, ex: F_25-34_0L
#personality_in_series = False # Set this to True in order to perform classification of author gender, age and one personality trait at a time
