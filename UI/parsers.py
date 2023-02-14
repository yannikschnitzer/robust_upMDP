import sys
import Models.test as test
import Models.converter as convert
import Markov.storm_interface as storm_ui

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

def parse_model(flag, opts):
    model_name = parse_str(flag, opts)
    if model_name == "test":
        model = test.get_model()
    elif model_name == "test2":
        model = test.get_model_2()
    elif model_name == "test3":
        model = test.get_model_3()
    elif model_name == "drone":
        prefix = "Models/benchmarks/drone/drone_"
        params, storm_model, props = storm_ui.load_problem(prefix+"model.nm",prefix+"spec.prctl", "none")
        model = convert.parse(storm_model, params, prefix+"model.nm", props, "uniform")
    return model

def parse_bool(flag):
    return True
