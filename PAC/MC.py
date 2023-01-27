
def sampler(model, k, N, thresh, violation_prob, pol=None):
    out_count = 0
    for i in range(N):
        inn_count = 0
        for j in range(k):
            if pol is None:
                sample = model.sample_MDP()
                IO = writer.stormpy_io(sample)
                IO.write()
                #IO.solve_PRISM()
                res, all_res = IO.solve()
                if res <= thresh:
                    inn_count+= 1
        inn_prob = inn_count/k
        if inn_prob > violation_prob:
            out_count += 1
    out_prob = out_count/N
    return out_prob, inn_prob
