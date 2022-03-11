import numpy as np

def get_all_combinations(bits=3):
    X = np.zeros((2**bits, bits)).astype(np.int)

    X[1, -1] = 1
    for i in xrange(2, bits+1):
        X[2**(i-1): 2**i, -i] = 1
        X[2**(i-1): 2**i, -i+1:] = X[: 2**(i-1), -i+1:]

    return X

def get_binary_adder_numbers(bits=3):
    get_bin = lambda n, bits: map(lambda x: int(x), bin(n)[2:].zfill(bits))

    max_numbers = 2**(2*bits)

    X = np.zeros((max_numbers, 2*bits))
    T = np.zeros((max_numbers, bits+1))

    X[1, -1] = 1
    for i in xrange(2, bits*2+1):
        X[2**(i-1): 2**i, -i] = 1
        X[2**(i-1): 2**i, -i+1:] = X[: 2**(i-1), -i+1:]
    
    n = 2**bits-1
    for i in xrange(0, n):
        T[np.arange(i, i+i*2**bits+1, n)] = get_bin(i, bits+1)
        T[np.arange(2**(2*bits)-(n-i)*n+i, 2**(2*bits)-n+i+1, n)] = get_bin(n+i+1, bits+1)
    T[np.arange(n, n*2**bits+1, n)] = get_bin(n, bits+1)

    return X, T

plus_op = lambda a, b: a+b
sub_op = lambda a, b: a-b
mul_op = lambda a, b: a*b
def get_binary_numbers(bits=3, operator=plus_op):

    max_numbers = 2**(2*bits)

    X = np.zeros((max_numbers, 2*bits)).astype(np.uint)

    X[1, -1] = 1
    for i in xrange(2, bits*2+1):
        X[2**(i-1): 2**i, -i] = 1
        X[2**(i-1): 2**i, -i+1:] = X[: 2**(i-1), -i+1:]

    factors = 2**np.arange(bits-1, -1, -1).astype(np.uint)
    numbers_col_1 = np.sum(X[:, :bits]*factors, axis=1)
    numbers_col_2 = np.sum(X[:, bits:]*factors, axis=1)

    result = operator(numbers_col_1, numbers_col_2)%2**bits
    T = X[result, bits:].copy()
    
    # result = operator(numbers_col_1, numbers_col_2)%2**(bits*2)
    # T = X[result].copy()

    print("X:\n{}".format(X))
    print("T:\n{}".format(T))
    raw_input("Press ENTER to continue...")
    
    return X, T
