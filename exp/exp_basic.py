import os
import torch
from model import Bi_FI
from model import Tri_FI

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Bi_FI': Bi_FI,
            'Tri_FI': Tri_FI, # 将字符串 "Tri_FI" 与模型类关联
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
