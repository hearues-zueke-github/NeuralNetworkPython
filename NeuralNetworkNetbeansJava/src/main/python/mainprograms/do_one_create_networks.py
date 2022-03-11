#! /usr/bin/python2.7

import subprocess

for i in xrange(5):
    subprocess.call(["./create_network_folder.py", "create4types", "binadder", "5", "144_"+str(i), "145_"+str(i), "146_"+str(i), "147_"+str(i)])
