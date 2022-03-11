#! /usr/bin/python2.7

import getpass
import os
import re
import subprocess
import sys

import multiprocessing as mp

from multiprocessing import Process, Queue

username = getpass.getuser()
os.chdir("/home/{}/Documents".format(username))

os.system("rm -f load_machine_*")

if __name__ == "__main__":
    machine_numbers = list(xrange(150, 157))

    FNULL = open(os.devnull, 'wb')

    subprocs = [subprocess.Popen(["./get_load_machine.sh", "{}".format(machine_number)]) for machine_number in machine_numbers]

    for subproc in subprocs: subproc.wait()

    loads_str = []
    for machine_number in machine_numbers:
        with open("load_machine_{}.txt".format(machine_number), "rb") as fin:
            lines = fin.readlines()
            if len(lines) == 0:
                continue
            line = lines[0]
            search = re.search("load.*: (.*)\n", line)
            numbers = search.group(1)
            load_times = list(map(float, numbers.replace(", ", "!").replace(",", ".").split("!")))
            loads_str.append(("machine_number: {}".format(machine_number),
                              load_times))

    for load_str in loads_str:
        print("{}".format(load_str))
