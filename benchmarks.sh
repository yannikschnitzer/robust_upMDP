#python run.py --sg_itt 250 --init_step 1000 --model consensus --inst 2,2 -vo --file_out con_22.txt --tol 1e-4 --save_res consensus_22_res.txt -N 200 --to 3600 --MC --MC_p --sg_only --MC_samples 1000
#python run.py --sg_itt 250 --init_step 1000 --model consensus --inst 2,32 -vo --file_out con_232.txt --tol 1e-4 --save_res consensus_232_res.txt -N 200 --to 3600 --MC --MC_p --sg_only --MC_samples 1000
#python run.py --sg_itt 250 --init_step 1000 --model consensus --inst 4,2 -vo --file_out con_42.txt --tol 1e-4 --save_res consensus_42_res.txt -N 200 --to 3600 --MC --MC_p --sg_only --MC_samples 1000
#python run.py --sg_itt 250 --init_step 1000 --model consensus --inst 4,4 -vo --file_out con_44.txt --tol 1e-4 --save_res consensus_44_res.txt -N 200 --to 3600 --MC --MC_p --sg_only --MC_samples 1000

#python run.py --sg_itt 2500 --init_step 1000 --model brp --inst 256,5 -vo --file_out brp_25615.txt --tol 1e-4 --save_res brp_25615_res.txt -N 200 --to 3600 --MC --MC_p --sg_only --MC_samples 1000
#python run.py --sg_itt 2500 --init_step 1000 --model brp --inst 4096,5 -vo --file_out brp_40965.txt --tol 1e-4 --save_res brp_40965_res.txt -N 200 --to 3600 --MC --MC_p --sg_only --MC_samples 1000

python3 run.py --sg_itt 2500 --init_step 1000 --model sav --inst 2,6,2,2 -vo --file_out sav_2622.txt --tol 1e-4 --save_res sav_2622_res.txt -N 200 --to 3600 --MC --MC_p --sg_only --MC_samples 10
#python3 run.py --sg_itt 2500 --init_step 1000 --model sav --inst 2,100,10,10 -vo --file_out sav_21001010.txt --tol 1e-4 --save_res sav_21001010_res.txt -N 200 --to 3600 --MC --MC_p --sg_only --MC_samples 1000
#python run.py --sg_itt 2500 --init_step 1000 --model sav --inst 4,6,2,2 -vo --file_out sav_3622.txt --tol 1e-4 --save_res sav_4622_res.txt -N 200 --to 3600 --MC --MC_p --sg_only --MC_samples 1000
#python run.py --sg_itt 2500 --init_step 1000 --model sav --inst 4,10,3,3 -vo --file_out sav_41033.txt --tol 1e-4 --save_res sav_41033_res.txt -N 200 --to 3600 --MC --MC_p --sg_only --MC_samples 1000

#python run.py --sg_itt 2500 --init_step 1000 --model zeroconf --inst 2 -vo --file_out zeroconf_2.txt --tol 1e-4 --save_res zeroconf_2_res.txt -N 200 --to 3600 --MC --MC_p --sg_only --MC_samples 1000
#python run.py --sg_itt 2500 --init_step 1000 --model zeroconf --inst 5 -vo --file_out zeroconf_5.txt --tol 1e-4 --save_res zeroconf_5_res.txt -N 200 --to 3600 --MC --MC_p --sg_only --MC_samples 1000
