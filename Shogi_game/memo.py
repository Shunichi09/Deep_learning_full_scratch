# memo

import numpy as np

def func(A, B, C):
    return A

# A = [1, 2, 3]
A = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
B = np.array([[1, 2]])

# print(func(*A))

# print(A)

print(A[:, 1, :])

A[:, 1, :] = B

print(A)