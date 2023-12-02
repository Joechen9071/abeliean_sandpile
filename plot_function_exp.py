import matplotlib.pyplot as plt
import numpy as np
fig,ax = plt.subplots(1,2)
ax[0].imshow(np.random.random((20,20)))
ax[1].imshow(np.random.random((20,20)))
t = zip([1,1,1,1],[2,2,2,2])

for i,(t1,t2) in enumerate(t):
    print(t1,t2)