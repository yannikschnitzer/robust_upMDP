import sys

def get_val(flag):
    ind = sys.argv.index(flag)
    if ind+1 == len(sys.argv):
        raise Exception("Please select a value for flag \""+flag+"\"")
    else:
        return sys.argv[ind+1]

def parse_num(flag, num_type, min_val=-sys.maxsize, max_val=sys.maxsize):
    val = get_val(flag) 
    try:
         res=num_type(val)
         if res < min_val or res > max_val:
             raise ValueError
         else:
             return res
    except ValueError:
         raise Exception("Please select a valid number of samples for flag "+flag)

def parse_str(flag, opts):
    val = get_val(flag)
    if opts is not None:
        if val not in opts:
         raise Exception("Please make a valid selection for flag "+flag)
    return val

def parse_bool(flag):
    return True
