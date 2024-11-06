# MMD-OPT

Repository associated with our IEEE T-ASE journal submission.


https://github.com/user-attachments/assets/4333c450-b1d0-47e8-9003-1ad484f856b8


***Step 0*** Create two folders *data* and *stats* (in the root directory) with the following directory structure:
1. data
   - static
     - gaussian
     - bimodal
     - trimodal
   - dynamic
     - inlane_cut_in_low
     - inlane_cut_in_high
     - lane_change
2. stats
   - static
     - gaussian
     - bimodal
     - trimodal
   - dynamic
     - inlane_cut_in_low
     - inlane_cut_in_high
     - lane_change

## Synthetic Static Environment

***Step 1*** To configure the obstacle scenarios and to modify the parameters associated with different distributions you need to edit ```obs_data_generate_static.py```.
Specifically, the functions ```compute_noisy_init_state_gaussian, compute_noisy_init_state_bimodal, compute_noisy_init_state_trimodal``` contain the parameters associated with different distributions.
In order to modify the initial obstacle positions you need to modify the function ```compute_obs_data```.

***Step 2*** Run the following command from the root directory:
```
python3 main_mpc_static.py --num_exps <int> --num_reduced_set <list of ints> --num_obs <list of ints>
--costs <list of str> --noises <list of str>
```
where *costs* can be one or all of <**mmd_opt**, **cvar**, **saa**> and *noises* can be one or all of <**gaussian**, **bimodal**, **trimodal**>. The above command will run given number of experiments for default 200 obstacle configurations. This number can be changed by modifying *num_configs* variable in ```main_mpc_static.py```

***Step 3*** Once *Step 2* is complete, there will be data files in the corresponding folders in *data-> static-> gaussian,bimodal,trimodal*. Now we need to calculate the statistics for the collected data. Run the following command from the root directory:
```
python3 validation_static.py  --num_obs <from Step 2 > --num_reduced_set <from Step 2 > --noises <from Step 2>  --num_exps <from Step 2 >
```
This will store the statisitics in the corresponding locations in the *stats* folder
