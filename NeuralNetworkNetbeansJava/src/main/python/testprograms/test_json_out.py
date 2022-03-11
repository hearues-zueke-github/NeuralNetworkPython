#! /usr/bin/python2.7
# coding: utf-8

import sys
import os
import json

from random import randint, random

def print_local(name, env):
    print(name+" = "+str(env[name]))

def main(argv):
    filepath = os.getcwd()
    filename = "dumpfile.json"
    print_local("filepath", locals())
    json_data = []
    for _ in xrange(0, 10):
        data = [[random() for i in xrange(0, 28)] for j in xrange(0, 28)]
        json_data.append(data)
    for jd in json_data:
        print(str(jd))
    json.dump(json_data, open(filepath+"/../../../../jsonfiles/"+filename, "w"))

if __name__ == "__main__":
    main(sys.argv)
