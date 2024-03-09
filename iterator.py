import subprocess
import time

EPOCHS = 50
num_iq_samples = 1024
codebook_slots = 256
num_res_blocks = [2,3]
KL_coeff = [0.1,0.001]
CL_coeff = [0.005,0.001]
Cos_coeff = [0.7,0.0001,0]
batch_norm = [0,1,2]
codebook_init = ['normal','uniform']
reset_choice = [0,1]
cos_reset = [0,1]

argument_sets = []

argument_sets.append(f"--EPOCHS 100 --num_iq_samples 1024 --layers 5 --codebook_slots 64 --codebook_dim 256 --num_res_blocks 2 --KL_coeff 0.1 --CL_coeff 0.005 --Cos_coeff 0.7 --batch_norm 1 --codebook_init normal --reset_choice 1 --cos_reset 0 --version 0 --compress 2 ")



print(len(argument_sets))
for arguments in argument_sets: 
    command1 = f"python hae_sig_main.py {arguments}"
    command2 = f"python hqa_sig_main.py {arguments}" 
    command3 = f"python Eff_classifier.py {arguments}"
    subprocess.run(command1, shell=True)
    subprocess.run(command2, shell=True)
    subprocess.run(command3, shell=True)
    ##print(arguments)

    
    pass
    # Sleep for some time before running the script again (adjust as needed)
    time.sleep(60)  # Sleep for 60 seconds between runs
 