import numpy as np
import os

out = np.empty([10,3], dtype=np.int32)

for i in range(10):
    train_count = len(np.load(os.path.join("discrete", str(i), "train.npy")))
    validation_count = len(np.load(os.path.join("discrete", str(i), "validation.npy")))
    test_count = len(np.load(os.path.join("discrete", str(i), "test.npy")))
    # extra_count = len(np.load(os.path.join("discrete", str(i), "extra.npy")))

    out[i, :] = train_count, validation_count, test_count

# out = np.empty([1, 3], dtype=np.int32)
# train_count = len(np.load(os.path.join("discrete", "combined", "train_data.npy")))
# validation_count = len(np.load(os.path.join("discrete", "combined", "validation_data.npy")))
# test_count = len(np.load(os.path.join("discrete", "combined", "test_data.npy")))
#
# out[0, :] = train_count, validation_count, test_count

# np.save(os.path.join("dataset_size"), out)
np.savetxt(os.path.join("dataset_size.txt"), out, fmt='%d')

