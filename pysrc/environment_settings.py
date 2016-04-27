import os, sys
# This file holds machine-specific paths and settings. It gets imported by weka_classify.py.

# for home/work machine:
javahome = None
javaw_path = None
wekahome = None

if sys.platform == 'win32':
	for possible_javahome in ('C:\\Program Files\\Java\\jdk1.8.0_60\\bin', 'C:\\Program Files\\Java\\jre7\\bin',	'C:\\Program Files\\Java\\jdk1.7.0_10\\jre\\bin', 'C:\\Program Files\\Java\\jre1.8.0_40\\bin'):
		if os.path.exists(possible_javahome):
			javahome = possible_javahome
			javaw_path = "{0}\\{1}".format(javahome, 'javaw.exe')
			break

	for possible_wekahome in ('C:\\Program Files\\Weka-3-7\\weka.jar', 'C:\\Weka-3-7\\weka.jar'):
		if os.path.exists(possible_wekahome):
			wekahome = possible_wekahome
			break
else:
	# for tira virtual machine:
	javahome = '/usr/bin/java'
	wekahome = '/home/mccollister15/weka-3-7-12/weka.jar'
	javaw_path = None
