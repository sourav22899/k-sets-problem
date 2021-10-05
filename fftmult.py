from cmath import exp
from math import pi
import numpy as np
from numpy.fft import rfft, irfft


def fftrealpolymul(arr_a, arr_b):  # fft based real-valued polynomial multiplication
    L = len(arr_a) + len(arr_b)
    a_f = rfft(arr_a, L)
    b_f = rfft(arr_b, L)
    return irfft(a_f * b_f)[:L-1]


"""
    Quick convolution that can interchangeably use both FFT or NTT, see example at bottom.
    This uses a somewhat advanced implementation of Cooley-Tukey that hopefully runs quickly with high
    numerical stability. /pajenegod
    
    Reference: https://discuss.codechef.com/t/help-in-polynomial-multiplication/19088/3
"""


def isPowerOfTwo(n):
    return n > 0 and (n & (n - 1)) == 0


# Permutates A with a bit reversal
# Ex. [0,1,2,3,4,5,6,7]->[0,4,2,6,1,5,3,7]
def bit_reversal(A):
    n = len(A)
    assert (isPowerOfTwo(n))

    k = 0
    m = 1
    while m < n: m *= 2;k += 1

    for i in range(n):
        I = i
        j = 0
        for _ in range(k):
            j = j * 2 + i % 2
            i //= 2
        if j > I:
            A[I], A[j] = A[j], A[I]
    return


"""
    NTT ALGORITHM BASED ON COOLEY TUKEY
    
    Inplace NTT using Cooley-Tukey, a divide and conquer algorithm
    running in O(n log(n)) time implemented iteratively using bit reversal,
    NOTE that Cooley-Tukey requires n to be a power of two
    and also that n <= longest_conv, basically
    n is limited by the ntt_prime
"""

# Remember to set ntt_prime and ntt_root before calling, for example
ntt_prime = (479 << 21) + 1
ntt_root = 3


def NTT_CT(A, inverse=False):
    # Some pre-calculations needed to do the ntt
    non_two = ntt_prime - 1
    longest_conv = 1
    while (ntt_prime - 1) % (2 * longest_conv) == 0: longest_conv *= 2
    ntt_base = pow(ntt_root, (ntt_prime - 1) // longest_conv, ntt_prime)

    N = len(A)
    assert (isPowerOfTwo(N))
    assert (N <= longest_conv)

    for i in range(N):
        A[i] %= ntt_prime

    # Calculate the twiddle factors
    e = pow(ntt_base, longest_conv // N, ntt_prime)
    if inverse:
        e = pow(e, ntt_prime - 2, ntt_prime)
    b = e
    twiddles = [1]
    while len(twiddles) < N // 2:
        twiddles += [t * b % ntt_prime for t in twiddles]
        b = b ** 2 % ntt_prime

    bit_reversal(A)

    n = 2
    while n <= N:
        offset = 0
        while offset < N:
            depth = N // n
            for k in range(n // 2):
                ind1 = k + offset
                ind2 = k + n // 2 + offset
                even = A[ind1]
                odd = A[ind2] * twiddles[k * depth]

                A[ind1] = (even + odd) % ntt_prime
                A[ind2] = (even - odd) % ntt_prime

            offset += n
        n *= 2

    if inverse:
        inv_N = pow(N, ntt_prime - 2, ntt_prime)
        for i in range(N):
            A[i] = A[i] * inv_N % ntt_prime
    return


"""
FFT ALGORITHM BASED ON Cooley-Tukey

Inplace FFT using Cooley-Tukey, a divide and conquer algorithm
running in O(n log(n)) time implemented iteratively using bit_reversal,
NOTE that Cooley-Tukey requires n to be a power of two
"""


def FFT_CT(A, inverse=False):
    N = len(A)
    assert (isPowerOfTwo(N))

    # Calculate the twiddle factors, with very good numerical stability
    e = -2 * pi / N * 1j
    if inverse:
        e = -e
    twiddles = [exp(e * k) for k in range(N // 2)]

    bit_reversal(A)

    n = 2
    while n <= N:
        offset = 0
        while offset < N:
            depth = N // n
            for k in range(n // 2):
                ind1 = k + offset
                ind2 = k + n // 2 + offset
                even = A[ind1]
                odd = A[ind2] * twiddles[k * depth]

                A[ind1] = even + odd
                A[ind2] = even - odd

            offset += n
        n *= 2

    if inverse:
        inv_N = 1.0 / N
        for i in range(N):
            A[i] *= inv_N
    return A


# Circular convolution in O(nlog(n)) time
def circ_conv(A, B, FFT):
    assert (len(A) == len(B))
    n = len(A)

    A = list(A)
    B = list(B)
    FFT(A)
    FFT(B)

    C = [A[i] * B[i] for i in range(n)]
    FFT(C, inverse=True)
    return C


# Polynomial multiplication in O((n+m)log(n+m)) time
def conv(A, B, FFT):
    n = len(A)
    m = len(B)
    N = 1
    while N < n + m - 1:
        N *= 2
    A = A + [0] * (N - n)
    B = B + [0] * (N - m)
    C = circ_conv(A, B, FFT)
    return C[:n + m - 1]


# example

def multiply_by_ntt(A, B):
    # Set ntt prime
    ntt_prime = (119 << 23) + 1
    ntt_root = 3

    return conv(A, B, NTT_CT)


def multiply_by_fft(A, B):
    return conv(A, B, FFT_CT)


# for ntt in [False, True]:
#     # Switch between using ntt or ftt for convolution
#     if ntt:
#         # Set ntt prime
#         ntt_prime = (119 << 23) + 1
#         ntt_root = 3
#
#         print('Using NTT for convolution')
#         FFT = NTT_CT
#     else:
#         print('Using FFT for convolution')
#         FFT = FFT_CT
#
#     # Example
#     A = [1, 1, 2, 3, 4, 5, 6, 100]
#     print('A', A)
#     FFT(A)
#     print('FFT(A)', A)
#     FFT(A, inverse=True)
#     print('iFFT(FFT(A))', A)
#
#     # Multiply (1+2x+3x^2) and (2+3x+4x^2+5x^3)
#     A = [1, 2, 3]
#     B = [2, 3, 4, 5]
#     print('A =', A)
#     print('B =', B)
#     print('A * B =', conv(A, B))
#
A = [0.1, 0.1]
B = [0.1, 0.1]
print('A =', A)
print('B =', B)
print(multiply_by_fft(A, B))
print(multiply_by_ntt(A, B))


def nextPowerOf2(n):
    count = 0

    if n and not (n & (n - 1)):
        return n

    while n != 0:
        n >>= 1
        count += 1

    return 1 << count


def elementary_symmetric_polynomial(X, k=1):
    # pad X upto nearest power of 2 length
    X_ = np.zeros(nextPowerOf2(len(X)), dtype=np.float64)
    X_[:len(X)] = X
    W = np.ones_like(X_, dtype=np.float64)
    X_ = np.vstack((W, X_)).T

    K = X_.shape[0]
    while K > 1:
        # import pdb; pdb.set_trace()
        # print('='*50)
        # print(X_)
        X_temp = []
        for i in range(0, K, 2):
            x, y = list(X_[i]), list(X_[i + 1])
            # X_temp.append(multiply_by_fft(x, y)[:k+1])
            # X_temp.append(fftrealpolymul(x, y)[:k+1])
            X_temp.append(np.polymul(x, y)[:k+1])
        X_ = np.asarray(X_temp)
        # X_ = np.around(X_, decimals=10)
        K = K // 2

    return list(X_.real)

from timeit import default_timer as timer
from tqdm import tqdm

N, t = 5000, 0
for _ in tqdm(range(100)):
    A = np.random.random(N)
    start = timer()
    a = elementary_symmetric_polynomial(A, k=10)
    end = timer()
    t += (end - start)
print(t/100)
import pdb;pdb.set_trace()
