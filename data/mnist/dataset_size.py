import numpy as np
import os

out = np.empty([10, 3], dtype=np.int32)

for i in range(10):
    train_count = len(np.load(os.path.join("discrete", str(i), "train.npy")))
    validation_count = len(np.load(os.path.join("discrete", str(i), "validation.npy")))
    test_count = len(np.load(os.path.join("discrete", str(i), "test.npy")))

    out[i, :] = train_count, validation_count, test_count

# np.save(os.path.join("indices", "dataset_size"), out)
np.savetxt(os.path.join("dataset_size.txt"), out, fmt='%d')

