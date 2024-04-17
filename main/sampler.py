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
    sample_array = [
                    [0.9404609671356791, 0.8981776326384185],
                    [0.9389832926382342, 0.8688531847967902],
                    [0.7788434399690806, 0.7766300925600227],
                    [0.8655612875208566, 0.807913759226833],
                    [0.9128951254260605, 0.9446908388879199],
                    [0.9181904445042588, 0.8186386387017405],
                    [0.8728675186314087, 0.8039206577984]]
    
    samples = []

    for pair in sample_array:
        print("Pair:",pair)
        point = dict()
        i = 0
        for e in model.params:
            point[e] = stormpy.RationalRF(pair[i])
            print(e, pair[i])
            i += 1
        samples.append(point)
    #samples = [model.param_sampler() for j in range(N)]
    return samples

def get_samples(args):
    if args["sample_load_file"] is not None:
        samples = load_samples(args["sample_load_file"])
    else:
        samples = gen_samples(args["model"], args["num_samples"])
    if args["sample_save_file"] is not None:
        save_data(args["sample_save_file"], samples)
    return samples
