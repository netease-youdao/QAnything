from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetTrainer

train_factory = {
  'ctdet': CtdetTrainer,
  'ctdet_mid': CtdetTrainer,
  'ctdet_small': CtdetTrainer
}
