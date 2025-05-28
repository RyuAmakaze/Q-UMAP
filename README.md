# Q-UMAP

Quantum Mapping using UMAP

This repository provides a helper function for projecting complex-valued
features to two dimensions using the [`umap-learn`](https://github.com/lmcinnes/umap) library.
The key logic lives in [`complex_umap.py`](complex_umap.py).

## Usage

```
import numpy as np
from complex_umap import complex_umap_embedding

# Example complex-valued data of shape (n_samples, n_features)
data = np.array([[1 + 1j, 2 - 0.5j], [3 - 4j, 1.5 + 2j]])
embedding = complex_umap_embedding(data)
print(embedding.shape)  # -> (2, 2)
```

`complex_umap_embedding` automatically splits complex numbers into their real and
imaginary components so they can be processed by UMAP.
