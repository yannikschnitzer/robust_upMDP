python run.py --sg_itt 250 --init_step 1000 --model consensus --inst 2,2 -vo --file_out con_22.txt --tol 1e-4 --save_res consensus_22_res.txt -N 200 --to 3600 --MC --MC_p
python run.py --sg_itt 250 --init_step 1000 --model consensus --inst 2,32 -vo --file_out con_232.txt --tol 1e-4 --save_res consensus_232_res.txt -N 200 --to 3600 --MC --MC_p
python run.py --sg_itt 250 --init_step 1000 --model consensus --inst 4,2 -vo --file_out con_42.txt --tol 1e-4 --save_res consensus_42_res.txt -N 200 --to 3600 --MC --MC_p
python run.py --sg_itt 250 --init_step 1000 --model consensus --inst 4,4 -vo --file_out con_44.txt --tol 1e-4 --save_res consensus_44_res.txt -N 200 --to 3600 --MC --MC_p

python run.py --sg_itt 250 --init_step 1000 --model brp --inst 256,15 -vo --file_out brp_25615.txt --tol 1e-4 --save_res brp_25615_res.txt -N 200 --to 36000 --MC --MC_p
python run.py --sg_itt 250 --init_step 1000 --model brp --inst 4096,5 -vo --file_out brp_40965.txt --tol 1e-4 --save_res brp_40965_res.txt -N 200 --to 3600 --MC --MC_p

python run.py --sg_itt 250 --init_step 1000 --model sav --inst 6,2,2 -vo --file_out sav_622.txt --tol 1e-4 --save_res sav_622_res.txt -N 200 --to 36000 --MC --MC_p
python run.py --sg_itt 250 --init_step 1000 --model sav --inst 100,10,10 -vo --file_out sav_1001010.txt --tol 1e-4 --save_res sav_1001010_res.txt -N 200 --to 3600 --MC --MC_p
python run.py --sg_itt 250 --init_step 1000 --model sav --inst 10,3,3 -vo --file_out sav_1033.txt --tol 1e-4 --save_res sav_1033_res.txt -N 200 --to 3600 --MC --MC_p

python run.py --sg_itt 250 --init_step 1000 --model zeroconf --inst 2 -vo --file_out zeroconf_2.txt --tol 1e-4 --save_res zeroconf_2_res.txt -N 200 --to 3600 --MC --MC_p
python run.py --sg_itt 250 --init_step 1000 --model zeroconf --inst 5 -vo --file_out zeroconf_5.txt --tol 1e-4 --save_res zeroconf_5_res.txt -N 200 --to 3600 --MC --MC_p
