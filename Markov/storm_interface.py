import numpy as np

from tqdm import tqdm

import core.create_drone_prism_model as create_prism
import stormpy
import stormpy.core
import stormpy.logic
import stormpy.pars
import stormpy.examples
import stormpy.examples.files

def search(values, searchFor):
    '''
    Search for a value in a list of values
    '''
    
    for k in values:
        if searchFor in k:
            return k
    return None

def load_problem(model_file, property_file, bisimulation_type):
    '''
    Load a model in Storm, given a model and property file

    Parameters
    ----------
    model_file : String of the model file to load
    property_file : String of the property file to load
    bisimulation_type : Either 'strong', 'weak', or 'none'

    Returns
    -------
    parameters : Object with info about all parameters
    model : Storm model
    properties : Parsed property in Storm

    '''
    
    # Load and build model from file
    path = model_file
    prism_program = stormpy.parse_prism_program(path)
    print("\nBuilding model from {}".format(path))
    with open(property_file) as f:
        formula_str= " ".join([l.rstrip() for l in f])
    
    # Load properties
    properties = stormpy.parse_properties_for_prism_program(formula_str, 
                                                            prism_program)
    properties = stormpy.parse_properties(formula_str, prism_program)

    # Set model options
    options = stormpy.BuilderOptions([p.raw_formula for p in properties])
    #options.set_add_out_of_bounds_state()
    
    model = stormpy.build_sparse_parametric_model_with_options(prism_program, options)
    #model = stormpy.build_sparse_parametric_model(prism_program, properties)

    print((model.model_type))
    print("Model supports parameters: {}".format(model.supports_parameters))
    
    parameters = model.collect_probability_parameters()
    numstate=model.nr_states
    
    print("Number of states before bisim:",numstate)
    print ("Number of params before bisim:",len(parameters))
    
    # Set bisimulation for model
    if bisimulation_type == 'strong':
        
        print(" -- Perform strong bisimulation")
        model = stormpy.perform_bisimulation(model, properties, 
                                             stormpy.BisimulationType.STRONG)
    
    elif bisimulation_type == 'weak':
        
        print(" -- Perform weak bisimulation")
        model = stormpy.perform_bisimulation(model, properties, 
                                             stormpy.BisimulationType.WEAK)
    
    elif bisimulation_type == 'none':
        
        print(" -- Skip bisimulation")
        pass
    
    else:
        
        raise RuntimeError(
          "Invalid bisimulation type (should be 'strong', 'weak', or 'none').")
    
    parameters= model.collect_probability_parameters()
    parameters_rew = model.collect_reward_parameters()
    parameters.update(parameters_rew)
    
    if bisimulation_type != 'none':
        numstate=model.nr_states
        print("Number of states after bisim:",numstate)
        print ("Number of params after bisim:",len(parameters))
    
    print('\nModel successfully loaded!\n')
    
    return parameters,model,properties,formula_str


def sample_MDP(parameters, model, model_file,
                   weather=None):
    '''
    Sample results for instantiated MDPs from Storm.

    Parameters
    ----------
    N : Number of parameter samples/solutions (integer)
    parameters : Parameters object
    model : Model object
    properties : Parsed properties in Storm
    model_file : String of the model file that is loaded
    weather : Weather conditions (optional; only used for drone benchmark)

    Returns
    -------
    solutions : List of solutions (one for each parameter sample)

    '''
    
    if 'drone' in model_file:
        drone = True
    else:
        drone = False
    
    # Evaluate model type
    if str(model.model_type)=='ModelType.DTMC':
        instantiator = stormpy.pars.PDtmcInstantiator(model)
    elif  str(model.model_type)=='ModelType.MDP':
        instantiator = stormpy.pars.PMdpInstantiator(model)
    else:
        raise RuntimeError("Invalid model type (should be a pDTMC or pMDP).")
    
    # If we are performing the drone benchmark
    if drone:
        groups = create_prism.parameter_groups()
        
        # Instantiate parameters in the model
        create_prism.parameter_definitions(groups,False)
            
    # Sample parameters according to the region defined by
    # "Parameter synthesis for Markov models: Faster than ever" paper
    point=dict()

    # Switch between drone and other benchmarks
    if drone:
        param_inst = create_prism.parameter_dirichlet_instantiations(
            groups, weather)
        param_samps = create_prism.parameter_dirichlet_samples(param_inst)
        
        for x in parameters:

        #instantiate parameters
            parameter_group = search(param_inst, str(x.name))

            element_int=0
            for element in parameter_group:
                
                param_sample_array = param_samps[tuple(parameter_group)]
                if (str(element)) == (str(x.name)):

                    point[x] = param_sample_array[0,element_int]
                element_int=element_int+1
    else:
        
        for x in parameters:
            if "coin" in model_file:
                s = np.random.uniform(0.2, 0.8)
    
            else:
                s = np.random.uniform(1e-5,1-1e-5)
    
            point[x] = stormpy.RationalRF(s)
    
    # Assign parameter values to model
    rational_parameter_assignments = dict(
        [[x, stormpy.RationalRF(val)] for x, val in point.items()])
    
    # Instantiate model
    inst_model = instantiator.instantiate(rational_parameter_assignments)
        
    return inst_model
