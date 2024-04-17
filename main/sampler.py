import pickle
import stormpy 

def load_samples(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data[0], data[1]

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def save_data(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def gen_samples(model, N):
    print("Num Samples:", N)
    sample_array = [[0.869661074152221, 0.8661798728546303],
                    [0.8628228067490115, 0.7951723834808841],
                    [0.8620823631217877, 0.8714771565000735],
                    [0.9125711960410308, 0.7818002418429729],
                    [0.8664964122259171, 0.8960449627795338],
                    [0.8207954625322589, 0.8478324770294388],
                    [0.8446490767938636, 0.7811642917262241],
                    [0.9191153719297983, 0.7936198699381651],
                    [0.8864324321419775, 0.9398784811009875],
                    [0.8072730371156094, 0.9214250405778299],
                    [0.8420246149869187, 0.8972084444641392],
                    [0.8142367177230883, 0.7645572236541063],
                    [0.8535789801814365, 0.9292351353230426],
                    [0.9404609671356791, 0.8981776326384185],
                    [0.9389832926382342, 0.8688531847967902],
                    [0.7788434399690806, 0.7766300925600227],
                    [0.8655612875208566, 0.807913759226833],
                    [0.9128951254260605, 0.9446908388879199],
                    [0.9181904445042588, 0.8186386387017405],
                    [0.8728675186314087, 0.8039202096577984]]
    
    samples = []

    for pair in sample_array:
        print("Pair:",pair)
        point = dict()
        i = 0
        for e in model.params:
            point[e] = stormpy.RationalRF(pair[i])
            print(e, pair[i])
            i += 1
        rational_parameter_assignments = dict(
            [[x, stormpy.RationalRF(val)] for x, val in point.items()])
        samples.append(rational_parameter_assignments)
    samples = [model.param_sampler() for j in range(N)]
    return samples

def get_samples(args):
    if args["sample_load_file"] is not None:
        samples = load_samples(args["sample_load_file"])
    else:
        samples = gen_samples(args["model"], args["num_samples"])
    if args["sample_save_file"] is not None:
        save_data(args["sample_save_file"], samples)
    return samples
