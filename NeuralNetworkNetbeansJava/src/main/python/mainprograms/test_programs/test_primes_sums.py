#! /usr/bin/python2.7

from __init__ import *

def get_primes(last_num):
    jumps = [4, 2]
    primes = [2, 3, 5]
    index = 0
    n = 7

    while n <= last_num:
        sqrt_n = int(np.sqrt(n))
        i = 2
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
        i = 2
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

def check_is_primes(primes, number):
    sqrt_num = int(np.sqrt(number))+1
    i = 0
    while primes[i] <= sqrt_num:
        if number % primes[i] == 0:
            return False
        i += 1
    return True

def show_digits_sum_diagram():
    primes = get_primes(20000)
    print("calculated primes = {}".format(len(primes)))
    # print("primes = {}".format(primes))

    digit_sum = [(p, np.sum(map(int, list(str(p))))) for p in primes]
    # print("digit_sum = {}".format(digit_sum))

    digit_sums_map = {}
    for (n, s) in digit_sum:
        if not s in digit_sums_map:
            digit_sums_map[s] = [n]
        else:
            digit_sums_map[s] += [n]
    # print("digit_sums_map = {}".format(digit_sums_map))

    keys = sorted(list(digit_sums_map.keys()))
    # print("keys = {}".format(keys))

    x_ax = keys
    y_ax = [len(digit_sums_map[key]) for key in keys]

    print("x_ax = {}".format(x_ax))
    print("y_ax = {}".format(y_ax))
    plt.plot(x_ax, y_ax, ".b")
    plt.show()

    # diffs = [p2-p1 for p2, p1 in zip(primes[1:], primes[:-1])]
    # print("diffs = {}".format(diffs))

def calc_primes_multiprocess(jumps=10000, iterations=5):
    # jumps = 10000
    last_num = jumps
    next_part = 1
    primes = get_primes(jumps)
    # print("Calculated primes to the number {}".format(jumps))

    processes_amount = mp.cpu_count()
    job_queues = [mp.Queue() for _ in xrange(processes_amount)]
    result_queues = [mp.Queue() for _ in xrange(processes_amount)]
    processes = [mp.Process(target=calc_next_primes, args=(i, job_queues[i], result_queues[i])) for i in xrange(processes_amount)]
    for proc in processes: proc.start()

    for i in xrange(0, processes_amount):
        job_queues[i].put([primes, next_part, jumps*next_part+1, jumps*(next_part+1)])
        next_part += 1

    # iterations = 5
    for j in xrange(iterations):
        # print("Iterations {}".format(j))
        for i in xrange(0, processes_amount):
            results = result_queues[i].get()
            proc_num, next_primes, part_num = results
            primes += next_primes
            last_num = jumps*(next_part-processes_amount+1)
            # print("last num: {:10}   calculated primes: {}".format(last_num, len(primes)))
            if j!=iterations-1:
                job_queues[i].put([primes, next_part, jumps*next_part+1, jumps*(next_part+1)])
            else:
                job_queues[i].put("Finished")
            next_part += 1

    # for i in xrange(processes_amount): job_queues[i].put("Finished")
    for proc in processes: proc.join()

    return primes

def get_prime_jumps(first_primes_amount):
    primes = get_primes(100)



print("Test primes calculation:")
taken_time, primes = utils.test_time_with_return(get_primes, [8100000], {})
print("Taken time: {}".format(taken_time))
print("calculated primes = {}".format(len(primes)))

print("Test primes calculation with multiprocessing:")
taken_time, primes = utils.test_time_with_return(calc_primes_multiprocess, [100000, 10], {})
print("Taken time: {}".format(taken_time))
print("calculated primes = {}".format(len(primes)))
