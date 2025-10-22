import numpy as np
x= np.array([-1,-2,1,1,2])
mask = x<=0
out = x.copy()
out[mask] =0
print(out)