import numpy as np

def relu(z):
    a = max(0,z)
    cache = z
    return a, cache
