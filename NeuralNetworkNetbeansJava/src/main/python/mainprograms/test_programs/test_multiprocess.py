#! /usr/bin/python2.7
# coding: utf-8

# vdsfvdfsvsdfvdfsv-*- vdsvdsvcoding: utf-8 -*-

from random import randint
from multiprocessing import Process, Lock
from time import sleep

import sys
print sys.getdefaultencoding()

alpha_string = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"§%&/()=?"
# print(str())
# alpha_string = alpha_string.decode('ascii', 'ignore')
def get_random_name(length):
    strlen = len(alpha_string)
    name = ""

    for i in xrange(0, length):
        name += alpha_string[randint(0, strlen - 1)]
    # for

    return name
# def

def f(lock, proc_num, name):
    # with lock:
    lock.acquire()
    print("proc #: "+str(proc_num)+"     name: "+name)
    # with
    lock.release()

    for i in xrange(0, 1000):
        sleep(0.01)
        with lock:
            print("proc # "+str(proc_num)+" prints #"+str(i))
        # with
    # for
# def

if __name__ == '__main__':
    lock = Lock()

    with lock:
        pl = [Process(target=f, args=(lock, i, get_random_name(10))) for i in xrange(0, 600)]
        for p in pl: p.start()
    # with
    # for p in pl: p.join()
# if
