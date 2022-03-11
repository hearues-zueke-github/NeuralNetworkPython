#! /usr/bin/python2.7

from __init__ import *
# print sys.getdefaultencoding()

def get_primes(last_num):
    jumps = [4, 2]
    primes = [2, 3, 5]
    index = 0
    n = 7

    while n <= last_num:
        sqrt_n = int(np.sqrt(n))
        i = 0
        is_prime = True
        while primes[i] <= sqrt_n:
            if n%primes[i] == 0:
                is_prime = False
                break
            i += 1
        if is_prime:
            primes.append(n)

        n += jumps[index]
        index = (index+1) % 2

    return primes

def get_next_primes(primes, n, last_num):
    if n%2 == 0:
        n += 1
    if n%3 == 0:
        n += 2
    if n%3 == 2: # This will save 33% of searching new primes!
        jumps = [2, 4]
    else:
        jumps = [4, 2]

    found_primes = []
    index = 0
    while n <= last_num:
        sqrt_n = int(np.sqrt(n))
        i = 0
        is_prime = True
        while primes[i] <= sqrt_n:
            if n%primes[i] == 0:
                is_prime = False
                break
            i += 1
        if is_prime:
            found_primes.append(n)

        n += jumps[index]
        index = (index+1) % 2

    return found_primes

def calc_next_primes(proc_num, job_queue, result_queue):
    while True:
        get_in = job_queue.get()
        if get_in == "Finished":
            break
        primes, part_num, start_num, end_num = get_in
        next_primes = get_next_primes(primes, start_num, end_num)
        result_queue.put([proc_num, next_primes, part_num])

if __name__ == '__main__':
    fname = "calculated_primes.pkl.gz"
    if os.path.isfile(fname):
        with gzip.open(fname, "rb") as fin:
            jumps, last_num, next_part, primes = pkl.load(fin)
        print("Loaded #{} primes with the last number {}, next_part is {} and {} jumps".format(len(primes), last_num, next_part, jumps))
    else:
        jumps = 100000
        last_num = jumps
        next_part = 1
        primes = get_primes(jumps)
        print("Calculated primes to the number {}".format(jumps))
    processes_amount = 3
    job_queues = [mp.Queue() for _ in xrange(processes_amount)]
    result_queues = [mp.Queue() for _ in xrange(processes_amount)]
    processes = [mp.Process(target=calc_next_primes, args=(i, job_queues[i], result_queues[i])) for i in xrange(processes_amount)]
    for proc in processes: proc.start()

    for i in xrange(0, processes_amount):
        job_queues[i].put([primes, next_part, jumps*next_part+1, jumps*(next_part+1)])
        next_part += 1

    iterations = 5000
    for j in xrange(iterations):
        print("Iterations {}".format(j))
        for i in xrange(0, processes_amount):
            results = result_queues[i].get()
            proc_num, next_primes, part_num = results
            primes += next_primes
            job_queues[i].put([primes, next_part, jumps*next_part+1, jumps*(next_part+1)])
            last_num = jumps*(next_part+1)
            next_part += 1
            print("calculated primes: {}".format(len(primes)))

    for i in xrange(processes_amount): job_queues[i].put("Finished")
    for proc in processes: proc.join()

    with gzip.open(fname, "wb") as fout:
        pkl.dump([jumps, last_num, next_part, primes], fout)
