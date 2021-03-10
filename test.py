import numpy as np 

arr = np.array([0, 0, 0, 0, 0, 0])
print(np.all(arr == arr[0]) and arr[0] == 0)