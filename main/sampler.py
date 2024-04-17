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
    sample_array = [[0.853833457524235, 0.8487370069784823],
[0.7998104687405153, 0.8247050602428736],
[0.8977084045416687, 0.7569052963888274],
[0.8980146147823382, 0.8342922058170706],
[0.75486804341957, 0.8956742452328984],
[0.8459072966336929, 0.9243992880729172],
[0.8679385824476037, 0.8967934544317224],
[0.9292039869066168, 0.7641734824849227],
[0.7588389304543408, 0.8379111152598968],
[0.8936258170933787, 0.8624308629763897],
[0.7511046627176499, 0.9234593780286079],
[0.8918433175594039, 0.8602289329758033],
[0.8631754815724335, 0.7832087880451596],
[0.9185863763556245, 0.7864604439769155],
[0.7533895348116292, 0.769674691837904],
[0.7862459282066695, 0.923764601744448],
[0.7609046324976376, 0.900597570034703],
[0.8223942504197935, 0.7877159534515162],
[0.8314928054893017, 0.7890082024672798],
[0.8810025646616519, 0.9091650920949094],
[0.843131396732528, 0.8309277796313991],
[0.7507359748521163, 0.8277696658171186],
[0.7535402568851206, 0.8795420687067939],
[0.8165574760936192, 0.7541679365835634],
[0.8802913414238945, 0.8247277543892687],
[0.8849742604852946, 0.9389114602906297],
[0.8938434094426171, 0.9180982478615929],
[0.8186111715505855, 0.8981873157336221],
[0.7548312814055235, 0.9362752456227117],
[0.8423976367129835, 0.8895245393294949]]
    
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
            [[x, val] for x, val in point.items()])
        samples.append(rational_parameter_assignments)
    print("Hallo")
 #   samples = [model.param_sampler() for j in range(N)]
    return samples

def get_samples(args):
    if args["sample_load_file"] is not None:
        samples = load_samples(args["sample_load_file"])
    else:
        samples = gen_samples(args["model"], args["num_samples"])
    if args["sample_save_file"] is not None:
        save_data(args["sample_save_file"], samples)
    return samples
