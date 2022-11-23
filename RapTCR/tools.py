import time


def timed(myfunc):
    # Decorator to keep track of time required to run a function
    def timed(*args, **kwargs):
        start = time.time()
        result = myfunc(*args, **kwargs)
        end = time.time()
        print(f"Total time to run '{myfunc.__name__}': {(end-start):.3f}s")
        return result

    return timed
