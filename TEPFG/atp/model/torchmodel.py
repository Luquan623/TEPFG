import os,torch

from atp.model.model import ModelAbstract
import  numpy as np

class TorchModel(ModelAbstract):

    def save(self, filepath=None):
        if filepath is None:
            filepath = self.checkpoint_path + ".last_model_state"
        with open(filepath, 'wb') as fout:
            state = {'model': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'torch_random_state': torch.random.get_rng_state().numpy(),
                     'numpy_random_state':np.random.get_state()}
            torch.save(state, fout)


    def load(self, filepath=None):
        if filepath is None:
            filepath = self.checkpoint_path + ".last_model_state"
        if not os.path.isfile(filepath):
            return

        with open(filepath, 'rb') as fin:
            state = torch.load(fin)
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            random_state = torch.from_numpy(state['torch_random_state'])
            torch.random.set_rng_state(random_state)
            np.random.set_state(state['numpy_random_state'])