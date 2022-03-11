#! /usr/bin/python3

import multiprocessing as mp
import numpy as np

def proc_function(input_queue, output_queue):
    command, arguments = input_queue.get()
    proc_num = arguments[0]

    while True:
        command, arguments = input_queue.get()
        if command == "FINISHED":
            return
        elif command == "GET_RANDOM":
            output_queue.put("From Proc #{} you get the number: {}".format(proc_num, np.random.randint(0, 1000)))
        elif command == "CALC_POW":
            output_queue.put((arguments[0], arguments[1], arguments[0]**arguments[1]))

def main_function():
    cores = mp.cpu_count()
    print("cores: {}".format(cores))

    input_queues = [mp.Queue() for _ in range(cores)]
    output_queues = [mp.Queue() for _ in range(cores)]

    procs = [mp.Process(target=proc_function, args=(input_queues[i], output_queues[i])) for i in range(cores)]
    for i in range(cores):
        input_queues[i].put(("", (i, )))
    for proc in procs: proc.start()

    ## Multiprocessing calc forward ##
    for j in range(0, 1000):
        for i, input_queue in enumerate(input_queues): input_queue.put(("CALC_POW", (2, j*8+i+1000000)))
        for output_queue in output_queues:
            basis, exponent, num = output_queue.get()
            print("{}**{}".format(basis, exponent))#, len(str(num))))

    for input_queue in input_queues: input_queue.put(("FINISHED", 0))
    for proc in procs: proc.join()

main_function()
