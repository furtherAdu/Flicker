import numpy as np

till_timo_seed = 1234  # first two random arrs
adu_seed = 4321  # first random arr

np.random.seed(adu_seed)  # TODO: change seed when running Adu
num_subs = 3
# conds = np.repeat(np.arange(8,12.1,.5,),2)
conds = np.arange(8,12.1,.5,)
repeat = 2
ord = []

for j in range(num_subs):
    for i in range(repeat):
        np.random.shuffle(conds)
        ord.append(conds.copy())

ord = np.array(ord)
print(ord)