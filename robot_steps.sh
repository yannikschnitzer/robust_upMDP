python run.py --model robot -vo --file_out --tol 0 --sg_itt 5000 --save_res robot_const --MC --MC_p --save_figs --init_step 0.1 --step_exp 0 & 
(sleep 2; python run.py --model robot -vo --file_out --tol 0 --sg_itt 5000 --save_res robot_dec --MC --MC_p --save_figs --init_step 1 --step_exp 0.5)
