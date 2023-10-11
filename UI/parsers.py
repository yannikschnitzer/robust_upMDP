import logging
import sys
import Models.test as test
import Models.expander as expander
import Models.example as ex
import Models.converter as convert
import Markov.storm_interface as storm_ui
from os.path import exists
from os import remove as rm
import datetime

import fileinput
import shutil

def replace_line(file, search_line, replacement):
    for line in fileinput.input(file, inplace=True): 
        print(line.rstrip().replace(search_line, replacement))


def get_val(flag):
    try:
        ind = sys.argv.index(flag)
        if ind+1 == len(sys.argv):
            raise Exception("Please select a value for flag \""+flag+"\"")
        else:
            return sys.argv[ind+1]
    except ValueError:
        raise Exception("With your chosen options you must include a choice for flag \""+flag+"\"")

def parse_num(flag, num_type, min_val=-sys.maxsize, max_val=sys.maxsize):
    val = get_val(flag) 
    try:
         res=num_type(val)
         if res < min_val or res > max_val:
             raise ValueError
         else:
             return res
    except ValueError:
         raise Exception("Please select a valid number of type "+str(num_type)+" for flag "+flag+" with value between "+str(min_val)+" and "+str(max_val))

def parse_str(flag, opts):
    val = get_val(flag)
    if opts is not None:
        if val not in opts:
            opt_str = "\nValid options are:\n"
            for opt in opts:
                opt_str += "-"+opt+"\n"
            raise Exception("Please make a valid selection for flag "+flag+opt_str)
    return val

def parse_model(flag, opts):
    model_name = parse_str(flag, opts)
    edited_files = []
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
    elif model_name == "test7":
        model = test.get_model_7()
    elif model_name == "hol":
        model = ex.get_model()
    elif model_name == "robot":
        model = ex.get_robot()
    elif model_name == "expander":
        inst = parse_num("--inst", int, min_val = 1)
        model = expander.get_model(inst)
    else:
        wind = None
        from opts import inst_opts
        if model_name == "drone":
            prefix = "Models/benchmarks/drone/drone_"
            model_f = prefix + "model.nm"
            spec_f = prefix + "spec.prctl"
            wind= parse_str("--inst", inst_opts[model_name])
            #wind = "uniform"
            #wind = "x-neg-bias"
            #wind = "y-pos-bias"
            bisim = "none"
        else:
            inst = parse_str("--inst", inst_opts[model_name])
            if model_name == "consensus":
                split_inst = inst.split(",")
                prefix = "Models/benchmarks2/consensus/coin" + split_inst[0]
                model_f = prefix+".pm"
                spec_f = prefix+".prctl"
                if inst[0] == "2":
                    bisim="strong"
                else:
                    bisim="none"
                search = "const int K;"
                replace = search[:-1] +"=" +split_inst[1] + ";"
                edited_files.append(model_f)
                shutil.copyfile(model_f, model_f+".old")
                replace_line(model_f, search, replace)
                import pdb; pdb.set_trace()
        #elif model_name == "consensus_2":
        #    prefix = "Models/benchmarks/consensus/coin2"
        #    model_f = prefix+".pm"
        #    spec_f = prefix+".prctl"
        #    bisim="strong"
        #elif model_name == "consensus_4":
        #    prefix = "Models/benchmarks/consensus/coin4"
        #    model_f = prefix+".pm"
        #    spec_f = prefix+".prctl"
        #    bisim="none"
        params, storm_model, props, f = storm_ui.load_problem(model_f,spec_f, bisim)
        model = convert.parse(storm_model, params, model_f, props,f, wind)
        model.Name = model_name
    for f in edited_files:
        rm(f)
        shutil.move(f+".old", f)
    return model

def parse_bool(flag):
    return True

def parse_debug(flag):
    if flag == "-v":
        logging.basicConfig(level=logging.INFO)
    elif flag == "-d":
        logging.basicConfig(level=logging.DEBUG)
    else:
        start = datetime.datetime.now().isoformat().split('.')[0]
        fname = "console_out/" + start + 'output.txt'
        if flag == "-vo":    
            logging.basicConfig(level=logging.INFO, filename = fname)
        elif flag == "-do":
            logging.basicConfig(level=logging.DEBUG, filename = fname)



def parse_file(flag, load):
    filename = "store/"+parse_str(flag, None)
    if load:
        file_exists = exists(filename)
        if not file_exists:
            raise FileNotFoundError
    return filename
