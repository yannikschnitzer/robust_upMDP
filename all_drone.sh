python run.py --sg_itt 250 --init_step 1000 --model drone --inst uniform -vo --file_out --tol 1e-4 --save_res drone_uniform.txt -N 5000 --MC --MC_p --save_figs  
python run.py --sg_itt 250 --init_step 1000 --model drone --inst x-neg-bias -vo --file_out --tol 1e-4 --save_res drone_x_neg.txt -N 5000 --MC --MC_p --save_figs  
python run.py --sg_itt 250 --init_step 1000 --model drone --inst y-pos-bias -vo --file_out --tol 1e-4 --save_res drone_y_pos.txt -N 5000 --MC --MC_p --save_figs 
