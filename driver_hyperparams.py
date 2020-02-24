import os
import time

seed = 2589
stepsz_dis = [1e-4, 2e-4, 3e-4]#[1e-04, 5e-04, 9e-04]
stepsz_gen = [1e-5, 2e-5, 3e-5]#[1e-05, 5e-05, 9e-05]
base_command = "python train_fbfadam.py output --test --inception-score --seed %i " % seed
commands = [base_command+"-lrd %.2e -lrg %.2e" % (lrd, lrg) for (lrd, lrg) in zip(stepsz_dis, stepsz_gen)]

print commands
time.sleep(5)

for command in commands:
   os.system(command)
