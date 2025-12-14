from time import perf_counter
from functools import wraps

def timeit(f):
    
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        f(*args, **kwargs)
        end = perf_counter()
        print("perf counter: {}s".format(f.__name__))

    return wrapper
