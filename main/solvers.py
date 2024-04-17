from functools import partial
import numpy as np
import multiprocessing as mp
import cvxpy as cp
import Markov.writer as writer
import Markov.models
from PAC.funcs import *
import itertools
from tqdm import tqdm
import time
import copy
import logging
import os
import matplotlib.pyplot as plt
import pycarl
from main.sampler import *
import datetime
from scipy.special import softmax


class optimiser:
    parallel_test = False
    def __init__(self):
        self.risk_func = None
        self.max_time = 100
        pass
    def solve(self):
        pass
    
    def check_timeout(self, start):
        if time.perf_counter() - start > self.max_time:
            print("Timed out!")
            return True
        else:
            return False
    
    def call_stormpy(self, test_MDP, pol, all_samples):
        res_list = []
        all_res_list = []
        sol_pol_list = []
        for sample in all_samples:
            #if type(sample) is list:
            test_MDP.Transition_probs = sample
            #else:
            #    test_MDP = model.fix_params(sample)
            
            if pol is not None:
                test_model = test_MDP.fix_pol(pol)
            else:
                test_model = test_MDP
            
            IO = writer.stormpy_io(test_model)
            #IO = writer.PRISM_io(test_model)
            IO.write()
    
            res, all_res, sol_pol = IO.solve()

            res_list.append(res[0])
            all_res_list.append(all_res)
            sol_pol_list.append(sol_pol)
        return res_list, all_res_list, sol_pol_list

    def test_pol(self, model, samples, pol=None, paramed_models = None):
        true_probs = []
        pols = []
        if model.opt == "max":
            wc = 1
        else:
            wc = 0
        start = time.perf_counter()
        if type(model) is Markov.models.storm_upMDP:
            stormpy_partial = partial(self.call_stormpy, Markov.models.MDP(model.fix_params(samples[0]))
                                        , pol)
        else:
            stormpy_partial = partial(self.call_stormpy, model.fix_params(samples[0]), pol)
        if paramed_models is not None:
            args = [[paramed_model] for paramed_model in paramed_models]
        else:
            args = [[copy.copy(model.fix_params(sample).Transition_probs)] for sample in samples]
            #args = [[sample] for sample in samples]
        if self.parallel_test:
            with mp.Pool() as p:
                res = p.map(stormpy_partial, args)
        else:
            res = [stormpy_partial(arg) for arg in args]
        for elem in res:
            pols += elem[2]
            true_probs += elem[1]            
            if model.opt == "max":
                if wc>min(elem[0]):
                    wc = min(elem[0])
            else:
                if wc<max(elem[0]):
                    wc = max(elem[0])
        end = time.perf_counter()
        logging.debug("Time for testing: {:.4f}".format(end-start))
        true_probs = np.array(true_probs)
        return wc, true_probs, pols

class mixed_opt(optimiser):
    def find_all_pols(self, model):
        base_vec = [False for i in model.Actions]
        act_vecs = []
        
        print("--------------------\nFinding policies")
        for a in model.Actions:
            vec = copy.copy(base_vec)
            vec[a] = True
            act_vecs.append(vec)
        act_poss = [[copy.copy(act_vecs[a]) for a in model.Enabled_actions[s]] for s in model.States]
        print("Built list, now building full policies")
        pols = [np.array(pol) for pol in itertools.product(*act_poss)]
        return pols
    
    def build_init_payoff(self, samples, model):
        start = time.perf_counter()
        pols = self.find_all_pols(model)
        all_probs = []
        non_dommed_pols = []
        print("--------------------\nBuilding initial payoff matrix")
        best_wc = 0
        for j, pol in enumerate(tqdm(pols)):
            if self.check_timeout(start):
                return None, None
            # could also find best deterministic policy here
            probs = self.test_pol(model, samples, pol)[1]@model.rho.T
            probs =probs.flatten()
            if model.opt == "max":
                wc = min(probs)
                if wc > best_wc:
                    best_wc = wc
                    det_pol = pol
            else:
                wc = max(probs)
                if wc < best_wc:
                    best_wc = wc
                    det_pol = pol
                    
            dommed = False
            for i, elem in enumerate(all_probs):
                if model.opt == "max":
                    if np.all(probs < elem):
                        dommed = True
                    elif np.all(probs > elem):
                        all_probs.pop(i)
                        non_dommed_pols.pop(i)
                else:
                    if np.all(probs > elem):
                        dommed = True
                    elif np.all(probs < elem):
                        all_probs.pop(i)
                        non_dommed_pols.pop(i)
            if not dommed:
                all_probs.append(probs)
                non_dommed_pols.append(j)
        pols = [pols[i] for i in non_dommed_pols]
        payoffs = np.vstack(all_probs)
        return payoffs, pols
    
    def calc_payoff_mat(self, samples, model):
        payoffs, pols = self.build_init_payoff(samples, model)
        if payoffs is None:
            return None, None, None
    
        print("--------------------\nRemoving dominated samples")
        
        non_domed_samples = []
        for i in tqdm(range(len(samples))):
            if model.opt == "max":
                test_arr = np.all(payoffs[:,i][:,np.newaxis] > payoffs, axis=0)
            else:
                test_arr = np.all(payoffs[:,i][:,np.newaxis] < payoffs, axis=0)
            if not np.any(test_arr):
                non_domed_samples.append(i)
        non_domed_payoffs = payoffs[:, non_domed_samples]
        return non_domed_payoffs, pols, non_domed_samples

class det(mixed_opt):
    def __init__(self, args):
        self.max_time = args["timeout"]
        self.risk_func = calc_eps_nonconvex

    def solve(self, samples, model):
        print("--------------\nFinding best deterministic policy through Stackelberg")
        start = time.perf_counter()
        payoffs, pols, rel_samples = self.calc_payoff_mat(samples, model)
        if payoffs is None:
            return -1, None, None, None
        if model.opt != "max":
            payoffs = -payoffs
        min_vals = np.min(payoffs, axis=1)
        best = np.argmax(min_vals)
        pol = pols[best]
        val = min_vals[best]
        val_loc = np.argwhere(payoffs[best,:] == val)
        supps = np.argwhere(payoffs[:,val_loc] >= val) #(-1 to remove val==val)
        supps = supps[:,0].tolist()
        supps.pop(supps.index(best))
    
        supps_ub = len(supps)
        
        block_samples = []
        block_set = set()
        for supp in supps:
            poss_samples = np.argwhere(payoffs[supp,:] <= val)
            curr_samples = poss_samples.flatten().tolist()
            block_samples.append(curr_samples)
            block_set  = block_set.union(curr_samples)
        for i in range(len(block_set)):
            if self.check_timeout(start):
                return -1, None, None, None
            test_sets = list(itertools.combinations(block_set, i+1))
            for test in test_sets:
                hitter = True
                for elem in block_samples:
                    intersect = set(test).intersection(elem)
                    if len(intersect) == 0:
                        hitter = False
                        break
                if hitter:
                    hit_set = test
                    break
            if hitter:
                break
        supps = hit_set
    
        info = {"pols":pol, "all":(payoffs[best,:]).flatten(), "ids":rel_samples}
        if model.opt == "min":
            val = -val
    
        return val, pol, supps, info

class MNE(mixed_opt):
    def __init__(self, args, MNE_algo):
        self.MNE_algo = MNE_algo
        self.max_time = args["timeout"]
        self.tol = args["tol"]
        self.itts = args["FSP_itts"]
        self.risk_func = calc_eps_risk_complexity

    def solve(self, samples, model):
        payoffs, pols, rel_samples = self.calc_payoff_mat(samples, model)
        if payoffs is None:
            return -1, None, None, None
        if model.opt != "max":
            payoffs = -payoffs
        pol, val = self.MNE_algo(payoffs, self.max_time, self.itts)
        if pol is not None:
            info = {"pols": pols, "all":(pol[0]@payoffs).flatten(), "ids":rel_samples}
            pol = (pol[0], pols)
            supps = np.argwhere(pol[0] >= 1e-5)
        else:
            supps = samples 
            info = None
        if val is None:
            val = -1
        if model.opt != "max":
            val = -val
        return val, pol, supps, info

class bellman(optimiser):
    parallel_grad=False#True
    def __init__(self, args, quiet = False):
        self.max_iters = args["sg_itts"]
        self.tol = args["tol"]
        self.init_step = args["init_step"]
        self.step_exp = args["step_exp"]
        self.max_time = args["timeout"]
        self.quiet = quiet
        #supp_tol=0.05
        #supp_tol =55 0*tol
        self.supp_tol=0.05 #conservative but works
        
        self.risk_func = calc_eps_risk_complexity

    def solve(self, samples, model):
        start = time.perf_counter()
        if not self.quiet:
            print("--------------------\nStarting Bellman Policy Iteration")
       
        if self.quiet:
            def tqdm(item):
                return item
        else:
            from tqdm import tqdm
        sample_trans_probs = []
        for sample in samples:
            new_MDP = model.fix_params(sample)
            sample_trans_probs.append(copy.copy(new_MDP.Transition_probs))
    
        num_states = len(model.States)
        num_acts = len(model.Actions)

        pol = np.zeros((num_states, num_acts))
        
        wc_hist = []
        best_hist = []
        tic = time.perf_counter()
        
        for s in model.States:
            a = np.random.choice(model.Enabled_actions[s])
            pol[s,a] = 1
        wc, true_probs, _ = self.test_pol(model, samples, pol, paramed_models = sample_trans_probs)
        
        worst = np.argwhere(true_probs[:,model.Init_state]==wc)
        worst = np.random.choice(worst.flatten())
        toc = time.perf_counter()
        logging.debug("Time for finding worst case: {:.3f}s".format(toc-tic)) # This is also done every iteration, could be sped up but takes ~6/1500 the time
        best_worst_pol = self.test_pol(model, [samples[worst]])[2][0]
        #test_wc, test_probs, _ = self.test_pol(model, samples, best_worst_pol, paramed_models = sample_trans_probs)
        #test_worst = np.argwhere(test_probs[:,model.Init_state]==test_wc).flatten()
        #if worst in test_worst:
        #    info = {"hist":[test_wc], "all":test_probs[:, model.Init_state]}
        #    if not self.quiet:
        #        print("Worst case holds with deterministic policy, deterministic is optimal")
        #    return test_wc, best_worst_pol, test_worst, info
        #pol = 0.1*pol + 0.9*best_worst_pol # a nicer start point
        
        wc, true_probs, _ = self.test_pol(model, samples, pol, paramed_models = sample_trans_probs)
        worst = np.argwhere(true_probs[:,model.Init_state]==wc)
        worst = np.random.choice(worst.flatten())
    
        if model.opt == "max":
            best = 0
        else:
            best = 1
    
        best_pol = pol
        alternative_supps = set()
        for i in tqdm(range(self.max_iters)):
            if self.check_timeout(start):
                break
                #return -1, None, None, None
            time_start = time.perf_counter()
            
            old_pol = np.copy(pol)
            alternative_supps.add(worst)
            pol = self.find_pol(model, pol, samples[worst])
            #grad_norm = np.linalg.norm(grad, ord=np.inf)
             
            time_grads = time.perf_counter()-time_start
            #if time_grads > 2:
            #    import pdb; pdb.set_trace()
            logging.debug("Total time for finding policy: {:.3f}".format(time_grads))
    
            wc, true_probs, _ = self.test_pol(model, samples, pol, paramed_models = sample_trans_probs) # This is taking a little while? (like 1 min, could probably speed up)
            worst = np.argwhere(true_probs[:,model.Init_state]==wc)
            worst = np.random.choice(worst.flatten())
            wc_hist.append(wc)
            if model.opt == "max":
                if wc > best:
                    best = wc
                    best_pol = pol
            else:
                if wc < best:
                    best = wc
                    best_pol = pol
            best_hist.append(best)
            logging.info("Iteration: {}".format(i+1))
            logging.info("Current value: {:.6f}, with sample {}".format(wc, worst))
            #logging.info("Policy inf norm change: {:.3f}".format(np.linalg.norm(pol-old_pol, ord=np.inf)))
            if len(wc_hist) >= 2:
                change=abs(wc_hist[-2]-wc_hist[-1])
                logging.info("Value change: {:.6f}".format(change))
                if change < self.tol:
                    break
        final_time = time.perf_counter() - start
        print("Bellman Policy iteration took {:.3f}s".format(final_time))
        wc, true_probs, _ = self.test_pol(model, samples, best_pol, paramed_models = sample_trans_probs)
        if model.opt == "max":
            active_sg = np.argwhere(true_probs[:, model.Init_state] <= wc+self.supp_tol) # this tol is hard to tune...
        else:
            active_sg = np.argwhere(true_probs[:, model.Init_state] >= wc-self.supp_tol)
        info = {"hist":best_hist, "all":true_probs[:, model.Init_state]}
    
        return best, best_pol, active_sg, info

    def find_pol(self, model, pol, worst_sample):
        
        test_MDP = model.fix_params(worst_sample)
        nom_MC = test_MDP.fix_pol(pol)
        
        _, nom_MC_sol, _ = self.test_pol(model, [worst_sample], pol)
        nom_MC_sol = nom_MC_sol.flatten() 
        nom_MC_list = []
        s_list = []
        
        num_batches = mp.cpu_count() 
        batch_size = len(model.States)//num_batches+1
        acts_bool = [[True  if a in model.Enabled_actions[s] else False for a in model.Actions] for s in model.States]
        grad_partial = partial(self.pol_state, nom_MC_sol, test_MDP.Transition_probs, test_MDP.trans_ids, model.opt) 
        pre_grad = time.perf_counter()
        if len(model.States) <= 320:
            args = zip(acts_bool, model.States)
            
            if not self.parallel_grad:
                res = [grad_partial(arg) for arg in args]
            else:
                with mp.Pool() as p:
                    res = p.map(grad_partial, args)
        else:
            args_batched = [(acts_bool[x:x+batch_size],model.States[x:x+batch_size]) for x in range(0,len(acts_bool),batch_size) ]
            if not self.parallel_grad:
                res = [grad_partial(arg) for arg in args_batched]
            else:
                with mp.Pool() as p:
                    res = p.map(grad_partial, args_batched)#, chunksize=batch_size)

        del(acts_bool)
        res_unbatched = []
        for elem in res:
            res_unbatched += elem
        del(res)
        new_pol = np.vstack(res_unbatched)

        time_grad = time.perf_counter()-pre_grad
        logging.debug("time for grad: "+str(time_grad))
        pre_norming = time.perf_counter()
        
        return new_pol 

    def pol_state(self, nom_MC_sol, MDP_trans, MDP_ids, opt, args): #, actions_bool_set, s_set):
        actions_bool_set = args[0]
        s_set = args[1]
        pols = []
        if type(actions_bool_set[0]) is not list:
            actions_bool_set = [actions_bool_set]
            s_set = [s_set]
        
        for actions_bool, s in zip(actions_bool_set, s_set):
            if opt == "max":
                best = 0
            else:
                best = 1
            pol = np.zeros((1, len(actions_bool)))
            a_counter = 0
            for a_id, a in enumerate(actions_bool):
                if a:
                    tic = time.perf_counter()
                    
                    s_primes_sol = nom_MC_sol[MDP_ids[s][a_counter]]
                    s_probs = np.array(MDP_trans[s][a_counter])
                    curr_val = s_probs.T@s_primes_sol
                    # We could do SGD, by evaluating gradient wrt a single run, this might offer some speed up, but probably not much
                    if opt == "max":
                        if  curr_val >= best:
                            best = curr_val
                            pol[0,:] = 0
                            pol[0, a_id] = 1
                    else:
                        if curr_val <= best:
                            best = curr_val
                            pol[0,:] = 0
                            pol[0, a_id] = 1

                    a_counter += 1
            pols.append(pol)
        return pols


class subgrad(optimiser):
    parallel_grad=False#True
    def __init__(self, args, quiet = False):
        self.max_iters = args["sg_itts"]
        self.tol = args["tol"]
        self.init_step = args["init_step"]
        self.step_exp = args["step_exp"]
        self.max_time = args["timeout"]
        self.quiet = quiet
        #supp_tol=0.05
        #supp_tol =55 0*tol
        self.supp_tol=0.05 #conservative but works
        
        self.risk_func = calc_eps_risk_complexity

    def solve(self, samples, model):
        start = time.perf_counter()
        if not self.quiet:
            print("--------------------\nStarting subgradient descent")
       
        if self.quiet:
            def tqdm(item):
                return item
        else:
            from tqdm import tqdm
        sample_trans_probs = []
        for sample in samples:
            new_MDP = model.fix_params(sample)
            sample_trans_probs.append(copy.copy(new_MDP.Transition_probs))
    
        num_states = len(model.States)
        num_acts = len(model.Actions)

        num_start_points = 1 
        sol_pols = []
        sol_ress = []
        all_active = []
        alternative_supps = set()
        pol = np.zeros((num_states, num_acts))
        
        wc_hist = []
        best_hist = []
        tic = time.perf_counter()
        
        for s in model.States:
            for a in model.Enabled_actions[s]:
                pol[s,a] = 1/len(model.Enabled_actions[s])
        wc, true_probs, _ = self.test_pol(model, samples, pol, paramed_models = sample_trans_probs)
        
        if model.opt == "min":
            worst = np.argmax((true_probs@(model.rho.T)))
        else:
            worst = np.argmin((true_probs@(model.rho.T)))
        worst = np.random.choice(worst.flatten())
        toc = time.perf_counter()
        logging.debug("Time for finding worst case: {:.3f}s".format(toc-tic)) # This is also done every iteration, could be sped up but takes ~6/1500 the time
        best_worst_pol = self.test_pol(model, [samples[worst]])[2][0]
        #test_wc, test_probs, _ = self.test_pol(model, samples, best_worst_pol, paramed_models = sample_trans_probs)
        #test_worst = np.argwhere(test_probs[:,model.Init_state]==test_wc).flatten()
        #if worst in test_worst:
        #    info = {"hist":[test_wc], "all":test_probs[:, model.Init_state]}
        #    if not self.quiet:
        #        print("Worst case holds with deterministic policy, deterministic is optimal")
        #    return test_wc, best_worst_pol, test_worst, info
        pol = 0.1*pol + 0.9*best_worst_pol # a nicer start point
        
        wc, true_probs, _ = self.test_pol(model, samples, pol, paramed_models = sample_trans_probs)
        if model.opt == "min":
            worst = np.argmax((true_probs@(model.rho.T)))
        else:
            worst = np.argmin((true_probs@(model.rho.T)))
        worst = np.random.choice(worst.flatten())
    
        if model.opt == "max":
            best = 0
        else:
            best = 1
    
        best_pol = pol
        for i in tqdm(range(self.max_iters)):
            if self.check_timeout(start):
                break
                #return -1, None, None, None
            time_start = time.perf_counter()
            
            old_pol = np.copy(pol)
            step = self.init_step/((i+1)**self.step_exp)
            #step = 1/((i+1)**(1/2))
            #step = 0.1
            alternative_supps.add(worst)
            grad = self.find_grad(model, pol, samples[worst])
            #grad_norm = np.linalg.norm(grad, ord=np.inf)
             
            time_grads = time.perf_counter()-time_start
            #if time_grads > 2:
            #    import pdb; pdb.set_trace()
            logging.debug("Total time for finding gradients: {:.3f}".format(time_grads))
            if model.opt == "max":
                pol += step*grad
            else:
                pol -= step*grad 
    
            for s in model.States:
                fin = False
                if len(model.Enabled_actions[s]) <= 1:
                    pass
                else:
                    acts = model.Enabled_actions[s]
                    normal = np.ones(len(acts))/len(acts)**(1/2)
                    orig = np.ones(len(acts))/len(acts)
                    pol_s_a = pol[s][acts]
                    v = pol_s_a - orig
                    dist = np.dot(v,normal)
                    proj = pol_s_a - dist*normal
                    while True:
                        normal = np.ones(len(acts))/len(acts)**(1/2)
                        orig = np.ones(len(acts))/len(acts)
                        pol_s_a = pol[s][acts]
                        v = pol_s_a - orig
                        dist = np.dot(v,normal)
                        proj = pol_s_a - dist*normal
                        if np.all(proj > 0):
                            break
                        else:
                            pos = np.where(proj>0)[0]
                            acts = [acts[p] for p in pos]
                    pol[s] = 0
                    pol[s][acts] = proj
            time_proj = time.perf_counter()-time_start-time_grads
            logging.debug("Time for projection step: {:.3f}".format(time_proj))
            wc, true_probs, _ = self.test_pol(model, samples, pol, paramed_models = sample_trans_probs) # This is taking a little while? (like 1 min, could probably speed up)
            if model.opt == "min":
                worst = np.argmax((true_probs@(model.rho.T)))
            else:
                worst = np.argmin((true_probs@(model.rho.T)))
            wc_hist.append(wc)
            if model.opt == "max":
                if wc > best:
                    best = wc
                    best_pol = pol
            else:
                if wc < best:
                    best = wc
                    best_pol = pol
            best_hist.append(best)
            logging.info("Iteration: {}".format(i+1))
            logging.info("Current value: {:.6f}, with sample {}".format(wc, worst))
            #logging.info("Policy inf norm change: {:.3f}".format(np.linalg.norm(pol-old_pol, ord=np.inf)))
            if len(wc_hist) >= 2:
                change=abs(wc_hist[-2]-wc_hist[-1])
                logging.info("Value change: {:.6f}".format(change))
                if change < self.tol:
                    print("Change to small!")
                    break
        final_time = time.perf_counter() - start
        print("Subgradient Descent took {:.3f}s".format(final_time))
        wc, true_probs, _ = self.test_pol(model, samples, best_pol, paramed_models = sample_trans_probs)
        if model.opt == "max":
            active_sg = np.argwhere(true_probs@model.rho.T <= wc+self.supp_tol) # this tol is hard to tune...
        else:
            active_sg = np.argwhere(true_probs@model.rho.T >= wc-self.supp_tol)
        info = {"hist":best_hist, "all":true_probs@model.rho.T}
        return best, best_pol, active_sg, info
   
    def grad_state(self, nom_MC_sol, MDP_trans, MDP_ids, args): #, actions_bool_set, s_set):
        actions_bool_set = args[0]
        s_set = args[1]
        grads = []
        if type(actions_bool_set[0]) is not list:
            actions_bool_set = [actions_bool_set]
            s_set = [s_set]
        
        for actions_bool, s in zip(actions_bool_set, s_set):
            grad = np.zeros((1, len(actions_bool)))
            if sum(actions_bool) > 1:
                a_counter = 0
                for a_id, a in enumerate(actions_bool):
                    if a:
                        tic = time.perf_counter()
                        
                        s_primes_sol = nom_MC_sol[MDP_ids[s][a_counter]]
                        s_probs = np.array(MDP_trans[s][a_counter])
                        # We could do SGD, by evaluating gradient wrt a single run, this might offer some speed up, but probably not much
                        
                        grad[0, a_id] = s_probs.T@s_primes_sol

                        a_counter += 1
            grads.append(grad)
        return grads

    def find_grad(self, model, pol, worst_sample):
        grad = np.zeros_like(pol)
        norm = 0
        time_sum = 0
        
        test_MDP = model.fix_params(worst_sample)
        nom_MC = test_MDP.fix_pol(pol)
        
        _, nom_MC_sol, _ = self.test_pol(model, [worst_sample], pol)
        nom_MC_sol = nom_MC_sol.flatten() 
        nom_MC_list = []
        s_list = []

        trans_arr = np.zeros((len(nom_MC.States),len(nom_MC.States)))
        for s, (s_primes, probs) in enumerate(zip(nom_MC.trans_ids, nom_MC.Transition_probs)):
            trans_arr[s][s_primes] = probs
        
        init_vec = nom_MC.rho
        #init_vec = np.zeros((1,len(nom_MC.States)))
        ##init_vec[nom_MC.Init_state] = 1 # Might need to change this
        #init_vec = 0.1*np.ones((1,len(nom_MC.States)))/(len(nom_MC.States)-1)
        #init_vec[:,nom_MC.Init_state] = 0.9 # Hmmmm
        gamma = nom_MC.gamma
        eta_pi = init_vec@np.linalg.inv(np.identity(len(nom_MC.States))-gamma*trans_arr)
        eta_pi *= (1-gamma) 


        num_batches = mp.cpu_count() 
        batch_size = len(model.States)//num_batches+1
        acts_bool = [[True  if a in model.Enabled_actions[s] else False for a in model.Actions] for s in model.States]
        grad_partial = partial(self.grad_state, nom_MC_sol, test_MDP.Transition_probs, test_MDP.trans_ids) 
        pre_grad = time.perf_counter()
        if len(model.States) <= 320:
            args = zip(acts_bool, model.States)
            
            if not self.parallel_grad:
                res = [grad_partial(arg) for arg in args]
            else:
                with mp.Pool() as p:
                    res = p.map(grad_partial, args)
        else:
            args_batched = [(acts_bool[x:x+batch_size],model.States[x:x+batch_size]) for x in range(0,len(acts_bool),batch_size) ]
            if not self.parallel_grad:
                res = [grad_partial(arg) for arg in args_batched]
            else:
                with mp.Pool() as p:
                    res = p.map(grad_partial, args_batched)#, chunksize=batch_size)

        del(acts_bool)
        res_unbatched = []
        for elem in res:
            res_unbatched += elem
        del(res)
        grad = np.vstack(res_unbatched)
        
        grad = eta_pi.T*grad
        

        time_grad = time.perf_counter()-pre_grad
        logging.debug("time for grad: "+str(time_grad))
        pre_norming = time.perf_counter()
        grad /= np.linalg.norm(grad, ord="fro")
        time_norming = time.perf_counter()-pre_norming
        logging.debug("time for fro: "+str(time_norming))
        
        return grad

class interval(optimiser):

    def __init__(self, args):
        self.max_time = args["timeout"]
        self.risk_func = calc_eps_risk_complexity
    
    def solve(self, samples, model):
        print("--------------------\nStarting iMDP solver")
        print("hereeeeeeeeeeeeeee")
        new_model = model.make_max()
        del(model)
        model = new_model
        del(new_model)
        model.max_time = self.max_time
        iMDP = model.build_imdp(samples)
        if iMDP is not None:
            supports = iMDP.supports
            IO = writer.PRISM_io(iMDP)
            IO.write()
            from opts import prism_folder  
            res, all_res, pol = IO.solve(prism_folder=prism_folder)
            res, _, _ = self.test_pol(model, samples, pol)
        else:
            return None, None, None, None
        if model.switch_res:
            res = 1-res
        return res, pol, supports, None

class thom_base(optimiser):

    def calc_probs(self, model, samples):
        start = time.perf_counter()
        probs = []
        min_prob = 1
        discarded = 0
        for sample in samples: 
            if self.check_timeout(start):
                break
            test_MDP = model.fix_params(sample)
            #IO = writer.PRISM_io(sample)
            IO = writer.stormpy_io(test_MDP)
            IO.write()
            #IO.solve_PRISM()
            res, all_res, _ = IO.solve()
            probs += res
        return probs

class thom_relax(thom_base):
    def __init__(self, args):
        self.max_time = args["timeout"]
        self.rho = args["rho"]
        self.risk_func = calc_eps_risk_complexity

    def solve(self, samples, model):
        print("--------------------\nStarting relaxed problem solver based on Thom's work")
        probs = self.calc_probs(model, samples)
        res, relaxers = self.optimise(probs, model.opt)
        return res, None, np.argwhere(relaxers >= 0), None

    def optimise(self, probs, opt):
        N = len(probs)
        x_s = cp.Variable(1+N)
        c = -np.ones((N+1,1))*self.rho
        if opt == "max":
            c[0] = 1
            b = np.array([0]+probs)
            A = np.eye(N+1)
            A[:, 0] = -1
            A[0,0] = 1
            A = -A
            constraints = [A@x_s <= b, x_s >= 0]
        else:
            c[0] = -1
            b = np.array([-1]+probs)
            A = np.eye(N+1)
            A[:, 0] = 1
            A[0,0] = -1
            #A = -A
            constraints = [A@x_s >= b, x_s >= 0]

        objective = cp.Maximize(c.T@x_s)


        prob = cp.Problem(objective, constraints)
        result = prob.solve()

        etas = x_s.value[1:]
        tau = x_s.value[0]
        
        return tau, etas

class thom_discard(thom_base):
    def __init__(self, args): 
        self.max_time = args["timeout"]
        self.lamb = args["lambda"]
        if args["lambda"] is not None:
            self.risk_func = calc_eta_discard
        else:
            self.risk_func = calc_eps

    def solve(self, samples, model):
        print("--------------------\nStarting Thom's solver")
        probs = self.calc_probs(model, samples)
        min_prob, discarded = self.discard(probs, model.opt)
        return min_prob, None, discarded, None

    def discard(self, probs, opt):
        lambda_val = self.lamb
        if lambda_val is None:
            if opt == "max":
                opt_prob = min(probs)
            else:
                opt_prob = max(probs)
            discarded = [None] # nb: this form so that len(discarded) = 1 for later risk calc
        else:
            if opt == "max":
                undiscarded = [p_val for p_val in probs if p_val >= lambda_val]
                if len(undiscarded)>0:
                    opt_prob = min(undiscarded)
                else:
                    opt_prob = 0
            else:
                undiscarded = [p_val for p_val in probs if p_val <= lambda_val]
                if len(undiscarded)>0:
                    opt_prob = max(undiscarded)
                else:
                    opt_prob = 1
    
            discarded = [i for i,p in enumerate(probs) if p not in undiscarded]
        return opt_prob, discarded
