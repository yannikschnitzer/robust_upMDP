import logging
import sys
import Models.test as test
import Models.converter as convert
import Markov.storm_interface as storm_ui
from os.path import exists

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
    elif model_name == "test4":
        model = test.get_model_4()
    elif model_name == "test5":
        model = test.get_model_5()
    elif model_name == "test6":
        model = test.get_model_6()
    else:
        wind = None
        if model_name == "drone":
            prefix = "Models/benchmarks/drone/drone_"
            model_f = prefix + "model.nm"
            spec_f = prefix + "spec.prctl"
            wind = "uniform"
            bisim = "none"
        elif model_name == "consensus_2":
            prefix = "Models/benchmarks/consensus/coin2"
            model_f = prefix+".pm"
            spec_f = prefix+".prctl"
            bisim="strong"
        elif model_name == "consensus_4":
            prefix = "Models/benchmarks/consensus/coin4"
            model_f = prefix+".pm"
            spec_f = prefix+".prctl"
            bisim="none"
        elif model_name == "brp_256":
            prefix = "Models/benchmarks/brp/brp"
            model_f = prefix+"_256_5.pm"
            spec_f = prefix+".prctl"
            bisim="weak"
        elif model_name == "brp_16":
            raise NotImplementedError
        elif model_name == "brp_32":
            raise NotImplementedError
        elif model_name == "crowds_10":
            prefix = "Models/benchmarks/crowds/crowds"
            model_f = prefix+"_10_5.pm"
            spec_f = prefix+".prctl"
            bisim="weak"
        elif model_name == "crowds_15":
            prefix = "Models/benchmarks/crowds/crowds"
            model_f = prefix+"_15_7.pm"
            spec_f = prefix+".prctl"
            bisim="strong"
        elif model_name == "nand_10":
            prefix = "Models/benchmarks/nand/nand"
            model_f = prefix+"_10_5.pm"
            spec_f = prefix+".prctl"
            bisim="weak"
        elif model_name == "nand_25":
            prefix = "Models/benchmarks/nand/nand"
            model_f = prefix+"_25_5.pm"
            spec_f = prefix+".prctl"
            bisim="strong"
        params, storm_model, props, f = storm_ui.load_problem(model_f,spec_f, bisim)
        model = convert.parse(storm_model, params, model_f, props,f, wind)
    return model

def parse_bool(flag):
    return True

def parse_debug(flag):
    if flag == "-v":
        logging.basicConfig(level=logging.INFO)
    elif flag == "-d":
        logging.basicConfig(level=logging.DEBUG)

def parse_file(flag, load):
    filename = "store/"+parse_str(flag, None)
    if load:
        file_exists = exists(filename)
        if not file_exists:
            raise FileNotFoundError
    return filename
