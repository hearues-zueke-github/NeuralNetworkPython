#! /usr/bin/python2.7

import subprocess
import sys

name_prefix = ""
name_suffix = sys.argv[1]

# For test purpose only!
# subprocess.call(["./create_network_folder.py", "digit", "1000", "1000", name_suffix+"_sgd_sigmoid", "1", "bgd", "sigmoid"])

# subprocess.call(["./create_network_folder.py", "digit", "1000", "1000", name_suffix+"_sgd_sigmoid", "10", "sgd", "sigmoid"])#, "cont"])
# subprocess.call(["./create_network_folder.py", "digit", "1000", "1000", name_suffix+"_sgd_tanh", "10", "sgd", "tanh"])#, "cont"])
# subprocess.call(["./create_network_folder.py", "digit", "1000", "1000", name_suffix+"_bgd_sigmoid", "10", "bgd", "sigmoid"])#, "cont"])
# subprocess.call(["./create_network_folder.py", "digit", "1000", "1000", name_suffix+"_bgd_tanh", "10", "bgd", "tanh"])#, "cont"])

#subprocess.call(["./create_network_folder.py", "adder", "5", "sgd_sigmoid_"+name_suffix, "10", "sgd", "sigmoid"])
# subprocess.call(["./create_network_folder.py", "adder", "5", name_suffix+"_sgd_tanh", "50", "sgd", "tanh"])
# subprocess.call(["./create_network_folder.py", "adder", "5", name_suffix+"_bgd_sigmoid", "50", "bgd", "sigmoid"])
# subprocess.call(["./create_network_folder.py", "adder", "5", name_suffix+"_bgd_tanh", "50", "bgd", "tanh"])
#
# subprocess.call(["./create_network_folder.py", "adder", "6", name_suffix+"_sgd_sigmoid", "20", "sgd", "sigmoid"])
# subprocess.call(["./create_network_folder.py", "adder", "6", name_suffix+"_sgd_tanh", "20", "sgd", "tanh"])
# subprocess.call(["./create_network_folder.py", "adder", "6", name_suffix+"_bgd_sigmoid", "20", "bgd", "sigmoid"])
# subprocess.call(["./create_network_folder.py", "adder", "6", name_suffix+"_bgd_tanh", "20", "bgd", "tanh"])

subprocess.call(["./create_network_folder.py", "addermany", "5", "noadapt_random_0_005_"+name_suffix, "10", "0", "0", "0", "0.005", "3"])
subprocess.call(["./create_network_folder.py", "addermany", "5", "adapt_random_0_005_"+name_suffix, "10", "0", "1", "0", "0.005", "3"])
subprocess.call(["./create_network_folder.py", "addermany", "5", "adapt_norandom_0_005_"+name_suffix, "10", "0", "0", "1", "0.005", "3"])
subprocess.call(["./create_network_folder.py", "addermany", "5", "noadapt_norandom_0_005_"+name_suffix, "10", "0", "1", "1", "0.005", "3"])

subprocess.call(["./create_network_folder.py", "addermany", "5", "noadapt_random_0_05_"+name_suffix, "10", "0", "0", "0", "0.05", "3"])
subprocess.call(["./create_network_folder.py", "addermany", "5", "adapt_random_0_05_"+name_suffix, "10", "0", "1", "0", "0.05", "3"])
subprocess.call(["./create_network_folder.py", "addermany", "5", "adapt_norandom_0_05_"+name_suffix, "10", "0", "0", "1", "0.05", "3"])
subprocess.call(["./create_network_folder.py", "addermany", "5", "noadapt_norandom_0_05_"+name_suffix, "10", "0", "1", "1", "0.05", "3"])

subprocess.call(["./create_network_folder.py", "addermany", "5", "noadapt_random_0_5_"+name_suffix, "10", "0", "0", "0", "0.5", "3"])
subprocess.call(["./create_network_folder.py", "addermany", "5", "adapt_random_0_5_"+name_suffix, "10", "0", "1", "0", "0.5", "3"])
subprocess.call(["./create_network_folder.py", "addermany", "5", "adapt_norandom_0_5_"+name_suffix, "10", "0", "0", "1", "0.5", "3"])
subprocess.call(["./create_network_folder.py", "addermany", "5", "noadapt_norandom_0_5_"+name_suffix, "10", "0", "1", "1", "0.5", "3"])
