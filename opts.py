from UI.parsers import *
import Models.test as test

opt_settings = {"model":{"parser":parse_model, 
                         "args":[["test", "test2", "test3", "test4","drone"]], 
                         "flags":["--model"],
                         "default":test.get_model_2()
                         },
                "num_samples":{
                    "parser":parse_num, 
                    "args":[int,1], 
                    "flags":["-N"],
                    "default":100
                    },
                "beta":{
                    "parser":parse_num,
                    "args":[float,0,1], 
                    "flags":["-beta"],
                    "default":0.99
                    },
                "lambda":{
                    "parser":parse_num,
                    "args":[float, 0, 1], 
                    "flags":["-lambda"],
                    "default":0
                    },
                "rho":{
                    "parser":parse_num,
                    "args":[float,0], 
                    "flags":["-rho"],
                    "default":1
                    },
                "MC":{
                    "parser":parse_bool,
                    "args":[], 
                    "flags":["--MC"],
                    "default":False
                    },
                "MC_samples":{
                    "parser":parse_num,
                    "args":[int, 1], 
                    "flags":["--MC_samples"],
                    "default":10000
                    },
                }
