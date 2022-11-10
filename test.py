import random
list_s1 = [1, 2, 3, 4]
n_train = int(len(list_s1)*0.8)
print(f'{n_train} - {len(list_s1)}')
random_idx = random.sample((0, len(list_s1)), n_train)
print(random_idx)
