
def run_all(args, samples):
    start = datetime.datetime.now().isoformat().split('.')[0]
    print("Running code for robust optimal policy \n --------------------")
    model = args["model"]
    if args["test_supps"]:
        test_support_num(args)
    else:
        a_priori_max_supports = model.max_supports
        a_priori_eps = calc_eps(args["beta"], args["num_samples"], a_priori_max_supports)
        
        print("A priori upper bound on number of support constraints is " + str(a_priori_max_supports))

        print("A priori bound on violation probability is {:.3f} with confidence {:.3f}".format(a_priori_eps, args["beta"]))
 
        if args["prob_load_file"] is not None:
            warm_probs = load_data(args["prob_load_file"])
        else:
            warm_probs = None
        num_states = len(model.States)

        ## subgradient

        N = args["num_samples"]
        start_time = time.perf_counter()
        res_sg, pol_sg, active_sg, info_sg = solve_subgrad(samples, model, max_iters=args["sg_itts"], tol=args["tol"], init_step=args["init_step"], step_exp=args["step_exp"])

        if pol_sg is not None: 
            sg_time = time.perf_counter()-start_time
            sg_active_num = active_sg.size 
            if args["save_figs"] or args["output_figs"]:
                res_plot = [res_sg - i for i in info_sg["hist"]]
                res_plot.pop(-1)
                fig, ax = plt.subplots()
                ax.loglog(res_plot)
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Distance from final satisfaction probability")
       

            if args["save_figs"]:
                fname = args["save_figs"] + '_dist_fig'
                plt.savefig(fname + ".png", bbox_inches="tight")
                plt.savefig(fname + ".pdf", bbox_inches="tight")
            elif args["output_figs"]:
                plt.show()
            
            if args["save_figs"] or args["output_figs"]:
                fig2, ax2 = plt.subplots()
                ax2.loglog(info_sg["hist"])
                ax2.set_xlabel("Iteration")
                ax2.set_ylabel("Satisfaction probability")
            
            if args["save_figs"]:
                fname = args["save_figs"] + '_prob_fig'
                plt.savefig(fname + ".png", bbox_inches="tight")
                plt.savefig(fname + ".pdf", bbox_inches="tight")
            elif args["output_figs"]:
                plt.show()
            
            if args["MC"]:
                emp_violation = MC_sampler(model, args["MC_samples"], res_sg, pol_sg) 
                print("Empirical violation rate is found to be {:.3f}".format(emp_violation))
            if args["MC_pert"]:
                pert_violation = MC_perturbed(model, args["MC_samples"], res_sg, pol_sg) 
                print("Noisy violation rate is found to be {:.3f}".format(pert_violation))

            print("Using subgradient methods found " + str(active_sg.size) + " active constraints a posteriori")
            [a_post_eps_L, a_post_eps_U] = \
                calc_eps_risk_complexity(args["beta"], N, active_sg.size)

            print("Hence, a posteriori, violation probability is in the range [{:.3f}, {:.3f}], with confidence {:.3f}"
                    .format(a_post_eps_L, a_post_eps_U, args["beta"]))

            print("Optimal satisfaction probability is found to be {:.3f}".format(res_sg))
        else:
            sg_time = -1
        pols = {"subgradient":pol_sg}
        res = {"subgradient":res_sg}

        MNE_time = -1
        FSP_time = -1
        det_time = -1
        if not args["sg_only"]:
            start_time = time.perf_counter()
            res_MNE, pol_MNE, MNE_support_num, info_MNE = MNE_solver(samples, model)
            if pol_MNE is not None:
                pol_MNE = (pol_MNE, info_MNE["pols"])
                MNE_time = time.perf_counter()-start_time
                if args["MC"]:
                    emp_MNE = MC_sampler(model, args["MC_samples"], res_MNE, pol_MNE) 
                    print("Empirical violation rate is found to be {:.3f}".format(emp_MNE))
                if args["MC_pert"]:
                    pert_MNE = MC_perturbed(model, args["MC_samples"], res_MNE, pol_MNE)
                    print("Noisy violation rate is found to be {:.3f}".format(pert_MNE))
                print("Using PNS algo found " + str(MNE_support_num) + " support constraints a posteriori")
                [MNE_a_post_eps_L, MNE_a_post_eps_U] = \
                    calc_eps_risk_complexity(args["beta"], N, MNE_support_num)

                print("Hence, a posteriori, violation probability is in the range [{:.3f}, {:.3f}], with confidence {:.3f}"
                    .format(MNE_a_post_eps_L, MNE_a_post_eps_U, args["beta"]))
                print("\n------------------\n")

            start_time = time.perf_counter()
            res_FSP, pol_FSP, FSP_a_post_support_num, info_FSP = FSP_solver(samples, model, max_iters=args["FSP_itts"])
            if pol_FSP is not None:
                pol_FSP = (pol_FSP, info_FSP["pols"])
                FSP_time = time.perf_counter()-start_time
                if args["MC"]:
                    emp_FSP = MC_sampler(model, args["MC_samples"], res_FSP, pol_FSP) 
                    print("Empirical violation rate is found to be {:.3f}".format(emp_FSP))
                if args["MC_pert"]:
                    pert_FSP = MC_perturbed(model, args["MC_samples"], res_FSP, pol_FSP)
                    print("Noisy violation rate is found to be {:.3f}".format(pert_FSP))
                [FSP_a_post_eps_L, FSP_a_post_eps_U] = \
                    calc_eps_risk_complexity(args["beta"], N, FSP_a_post_support_num)
                print("Using FSP algo found " + str(FSP_a_post_support_num) + " support constraints a posteriori")

                print("Hence, a posteriori, violation probability is in the range [{:.3f}, {:.3f}], with confidence {:.3f}"
                    .format(FSP_a_post_eps_L, FSP_a_post_eps_U, args["beta"]))
                print("\n------------------\n")

            start_time = time.perf_counter()
            res_det, pol_det, supps_det, info_det = det_maxmin(samples, model)
            if pol_det is not None:
                det_time = time.perf_counter()-start_time
                eps_det = calc_eps_nonconvex(args["beta"], N, supps_det)
                if args["MC"]:
                    emp_det = MC_sampler(model, args["MC_samples"], res_det, pol_det) 
                    print("Empirical violation rate is found to be {:.3f}".format(emp_det))
                if args["MC_pert"]:
                    pert_det = MC_perturbed(model, args["MC_samples"], res_det, pol_det)
                    print("Noisy violation rate is found to be {:.3f}".format(pert_det))
                print("Using deterministic policy found " + str(supps_det) + " support constraints a posteriori")

                print("Hence, a posteriori, violation probability is bounded by {:.3f}, with confidence {:.3f}"
                    .format(eps_det, args["beta"]))
                print("\n------------------\n")
            #if active_sg.size != a_post_support_num:
            #    print("Found {} supports using subgradient method, but {} using fictitious self play".format(active_sg.size, a_post_support_num))
            

            print("----------------\nResult comparison:\nPNS algo: {:.13f}\nDet.: {:.13f}\nFSP: {:.13f}\nSubgradient: {:.13f}".format(res_MNE, res_det, res_FSP, res_sg))
            res["MNE"] = res_MNE
            res["FSP"] = res_FSP
            res["det"] = res_det
            pols["MNE"] = pol_MNE
            pols["FSP"] = pol_FSP
            pols["det"] = pol_det

        if args["result_save_file"] is not None:
            save_data(args["result_save_file"], {"res": res, "pols":pols})
        if pol_sg is not None:
            if pol_sg.size < 50:
                print("Calculated robust policy using subgradient methods is:")
                print(pol_sg)

        thresh = a_priori_eps

        print("\n\n")
        return MNE_time, FSP_time, sg_time, det_time

