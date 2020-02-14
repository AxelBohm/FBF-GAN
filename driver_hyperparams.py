import os
import time

stepsz_dis = [1e-04, 5e-04, 9e-04]
stepsz_gen = [1e-05, 5e-05, 9e-05]
base_command = "python train_fbfadam.py output --test --inception-score "
commands = [base_command+"-lrd %.2e -lrg %.2e" % (lrd, lrg) for (lrd, lrg) in zip(stepsz_dis, stepsz_gen)]

print commands
time.sleep(5)

for command in commands:
   os.system(command)
