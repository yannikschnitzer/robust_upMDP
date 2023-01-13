import sys

def get_val(flag):
    ind = sys.argv.index(flag)
    if ind+1 == len(sys.argv):
        raise Exception("Please select a value for flag \""+flag+"\"")
    else:
        return sys.argv[ind+1]

def run():
    opts = {"test":False,
            "num_samples":100,
            "beta":0.99,
            "lambda":None,
            "rho":None,
            }
    if "--test" in sys.argv:
        opts["test"] = True
    if "-N" in sys.argv:
        val = get_val("-N")
        try:
            N=int(val)
            opts["num_samples"] = N
            if N <= 0:
                raise ValueError
        except ValueError:
            raise Exception("Please select an integer number of samples for flag -N")
    if "-beta" in sys.argv:
        val = get_val("-beta")
        try:
            int_val=float(val)
            if int_val > 1 or int_val < 0:
                raise ValueError
            opts["beta"] = int_val
        except ValueError:
            raise Exception("Please select a valid probability for flag -beta")
    if "-lambda" in sys.argv:
        val = get_val("-lambda")
        try:
            int_val=float(val)
            if int_val > 1 or int_val < 0:
                raise ValueError
            opts["lambda"] = int_val
        except ValueError:
            raise Exception("Please select a valid probability for flag -lambda")
    if "-rho" in sys.argv:
        val = get_val("-rho")
        try:
            int_val=float(val)
            if int_val > 1 or int_val < 0:
                raise ValueError
            opts["rho"] = int_val
        except ValueError:
            raise Exception("Please select a valid probability for flag -lambda")
 
    return opts
