import os
import time

seed = [1318, 2589, 3017, 9001]
# inertia = [0.0, 0.05, 0.1, 0.2, 0.3]
# stepsz_dis = [1e-04, 5e-04, 9e-04] #[1e-4, 2e-4, 3e-4]
# stepsz_gen = [1e-05, 5e-05, 9e-05] #[1e-5, 2e-5, 3e-5]
# base_command = "python train_tsengadam.py output --test --inception-score -bp --seed %i " % seed
# commands = [base_command+"-lrd %.2e -lrg %.2e" % (lrd, lrg) for (lrd, lrg) in zip(stepsz_dis, stepsz_gen)]
# base_command = "python train_fbfadam.py output --test --inception-score --seed %i " % seed
# base_command = base_command+"-lrd %.2e -lrg %.2e" % (stepsz_dis, stepsz_gen)
# commands = [base_command+" --inertia %.2f" % inert for inert in inertia]

# base_command = "python train_fbfadam.py output --test --inception-score "
# commands = []
# for s in seed:
#     for inert in inertia:
#         c = base_command + "--seed %i --inertia %.2f" % (s, inert)
#         commands.append(c)

# base_command = "python train_aybatadam.py output --cuda -gp 0 --model dcgan --default --inception-score "
# commands = [base_command+"--seed %i " % s for s in seed]

# stepsz_dis = [1e-04, 2e-04, 5e-04] #[1e-4, 2e-4, 3e-4]
# stepsz_gen = [1e-05, 2e-05, 5e-05] #[1e-5, 2e-5, 3e-5]

base_command = "python train_adam.py output --update-frequency 1 --default --inception-score --model dcgan -gp 0 --cuda -rp 0.0001 -lrd 2e-04 -lrg 2e-05 "
# base_command = "python train_optimisticadam.py output --default --inception-score --model dcgan -gp 0 --cuda -lrd 5e-04 -lrg 5e-05 -rp 0.0001 "
# commands = [base_command+"-lrd %.2e -lrg %.2e" % (lrd, lrg) for (lrd, lrg) in zip(stepsz_dis, stepsz_gen)]
commands = [base_command+"--seed %i " % s for s in seed]

print commands
time.sleep(3)

for command in commands:
   os.system(command)
