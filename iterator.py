import subprocess
import time

# List of experiments to run
argument_sets = [
    f"--EPOCHS 100 --num_iq_samples 1024 --layers 5 --codebook_slots 64 --codebook_dim 256 --num_res_blocks 2 --KL_coeff 0.1 --CL_coeff 0.005 --Cos_coeff 0.7 --batch_norm 1 --codebook_init normal --reset_choice 1 --cos_reset 0 --version 0 --compress 2 "
]

for arguments in argument_sets: 
    command1 = f"python hae_sig_main.py {arguments}"
    command2 = f"python hqa_sig_main.py {arguments}" 
    command3 = f"python Eff_classifier.py {arguments}"

    subprocess.run(command1, shell=True)
    subprocess.run(command2, shell=True)
    subprocess.run(command3, shell=True)
    
    time.sleep(60)  # Sleep for 60 seconds between runs
 