from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numba

@numba.jit(nopython=True, nogil=True)
def gen_oracle_map(feat, ind, w, h):
  # feat: B x maxN x featDim
  # ind: B x maxN
  batch_size = feat.shape[0]
  max_objs = feat.shape[1]
  feat_dim = feat.shape[2]
  out = np.zeros((batch_size, feat_dim, h, w), dtype=np.float32)
  vis = np.zeros((batch_size, h, w), dtype=np.uint8)
  ds = [(0, 1), (0, -1), (1, 0), (-1, 0)]
  for i in range(batch_size):
    queue_ind = np.zeros((h*w*2, 2), dtype=np.int32)
    queue_feat = np.zeros((h*w*2, feat_dim), dtype=np.float32)
    head, tail = 0, 0
    for j in range(max_objs):
      if ind[i][j] > 0:
        x, y = ind[i][j] % w, ind[i][j] // w
        out[i, :, y, x] = feat[i][j]
        vis[i, y, x] = 1
        queue_ind[tail] = x, y
        queue_feat[tail] = feat[i][j]
        tail += 1
    while tail - head > 0:
      x, y = queue_ind[head]
      f = queue_feat[head]
      head += 1
      for (dx, dy) in ds:
        xx, yy = x + dx, y + dy
        if xx >= 0 and yy >= 0 and xx < w and yy < h and vis[i, yy, xx] < 1:
          out[i, :, yy, xx] = f
          vis[i, yy, xx] = 1
          queue_ind[tail] = xx, yy
          queue_feat[tail] = f
          tail += 1
  return out