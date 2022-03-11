#! /usr/bin/python2.7
# coding: utf-8

import sys
import os
import json

from random import randint

def print_local(name, env):
    print(name+" = "+str(env[name]))

def main(argv):
    filepath = os.getcwd()
    filename = "dumpfile.json"

    json_data = json.load(open(filepath+"/../../../jsonfiles/"+filename, "r"))
    # print_local("json_data", locals())
    for jd in json_data:
        print(str(jd))

if __name__ == "__main__":
    main(sys.argv)
