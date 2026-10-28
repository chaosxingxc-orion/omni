"""Count all-zero vectors per prefix in encbench npz files (the left-padding / lasttoken pooling symptom)."""
import sys

import numpy as np

for path in sys.argv[1:]:
    data = np.load(path)
    for prefix in ("b1__", "b16__"):
        keys = sorted(k for k in data.files if k.startswith(prefix))
        zero = [k[len(prefix):] for k in keys if not np.any(data[k])]
        print(path.rsplit("/", 1)[-1], prefix, f"{len(zero)}/{len(keys)} zero", zero)
