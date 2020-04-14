import dlr
import numpy as np
import time
model = dlr.DLRModel('compiled', dev_type='gpu')
data = np.random.random((1, 3, 608, 608))
y = model.run(data)

times = []
for i in range(100):
  start = time.time()
  model.run(data)
  times.append(time.time() - start)

print('mean latency', np.mean(times[10:])*1000.0)
