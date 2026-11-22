import numpy as np

batch = [
    {"obs": np.array([[1, 2, 2], [3, 4, 5]])},
    {"obs": np.array([[5, 6, 5], [7, 8, 4]])},
]

result = {}

for key in batch[0]:
    values = []

    for item in batch:
        values.append(item[key])

    result[key] = np.stack(values)

print(result["obs"])
print(result["obs"].shape)