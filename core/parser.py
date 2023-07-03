import argparse
from ast import literal_eval

def parser():
    '''
    Parse arguments provided on the command line

    Returns
    -------
    args : Object containing raw argument values

    '''

    # Define the parser
    P = argparse.ArgumentParser(description='Sampler interfact')
    
    # Declare an argument (`--algo`), saying that the
    # corresponding value should be stored in the `algo`
    # field, and using a default value if the argument
    # isn't given
    P.add_argument('--model', action="store", dest='model', default=0)
    P.add_argument('--property', action="store", dest='property', default=0)
    P.add_argument('--bisimulation', action="store", dest='bisim', 
                   default='strong')
    P.add_argument('--threshold', action="store", dest='threshold', 
                   default=False)
    P.add_argument('--num_samples', action="store", dest='num_samples', 
                   default=1000)
    P.add_argument('--num_iter', action="store", dest='num_iter', default=1)
    P.add_argument('--comparator', action="store", dest='comparator', 
                   default=0)
    P.add_argument('--eta', action="store", dest='eta', default=0)
    P.add_argument('--beta', action="store", dest='beta', default=0)
    P.add_argument('--outfile', action="store", dest='outfile', 
                   default='output')

    # Weather conditions for drone    
    P.add_argument('--weather', action="store", dest='weather', default=0)
    
    # Now, parse the command line arguments and store the
    # values in the `args` variable
    args = P.parse_args()
    
    return args

def parse_settings():
    '''
    Do some basic checks and store argument values in a dictionary

    Returns
    -------
    Z : Dictionary of arguments

    '''
    
    # Parse the arguments provided
    args = parser()
    
    # Interpret some arguments as lists
    beta=args.beta
    if beta != 0:
        try:
            beta = [float(args.beta)]
        except:
            beta = list(literal_eval(args.beta))
            
    try:
        num_samples = [int(args.num_samples)]
    except:
        num_samples = list(literal_eval(args.num_samples))
        
    try:
        eta = [int(args.eta)]
    except:
        eta = list(literal_eval(args.eta))
    
    try:
        comparator = literal_eval(args.comparator)
    except:
        comparator = [args.comparator]
    
    if any([c != "leq" and c != "geq" for c in comparator]):
        raise RuntimeError("Invalid direction type: should be 'leq' or 'geq'")
        
    if type(args.threshold) is bool:
        threshold = False
    else:
        threshold = float(args.threshold)
        
    # Store arguments in dictionary
    Z = {
        'model': args.model,
        'property': args.property,
        'bisimulation': args.bisim,
        #
        'iterations': int(args.num_iter),
        'Nsamples': num_samples,
        'eta': eta,
        'beta': beta,
        #
        'threshold': threshold,
        'comparator': comparator,
        #
        'outfile': str(args.outfile),
        #
        'weather': args.weather
        }
        
    return Z