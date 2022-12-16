
def fixed(val):
    def func(_):
        return val
    return func

def linear(param):
    return min(max(param,0),1)

def one_minus_linear(param):
    return min(max(1-param,0),1)
