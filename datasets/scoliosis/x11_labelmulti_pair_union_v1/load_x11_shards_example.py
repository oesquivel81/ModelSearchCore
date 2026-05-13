import os
import glob
import numpy as np

ROOT = os.path.dirname(__file__)
SHARD_DIR = os.path.join(ROOT, 'shards_npz')

shards = sorted(glob.glob(os.path.join(SHARD_DIR, '*.npz')))
print('N shards:', len(shards))

assert len(shards) > 0, 'No se encontraron shards .npz'

data = np.load(shards[0], allow_pickle=True)

print('Keys:', list(data.keys()))
for k in data.keys():
    print(k, data[k].shape, data[k].dtype)

X = data['X']
y_multiclass = data['y_multiclass']
y_binary = data['y_binary']
y_boundary = data['y_boundary']
y_intervertebral = data['y_intervertebral']
y_ordinal = data['y_ordinal']

print('X:', X.shape)
print('channel_names:', data['channel_names'])
