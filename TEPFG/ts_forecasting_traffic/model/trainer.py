import io
import numpy as np
import pickle
import sys
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ts_forecasting_traffic.model.earlystopping import EarlyStopping


class TrainerBase:
    def __init__(self, epochs, evaluate_steps=1,
                 valid_on_train_set=False,
                 valid_on_test_set=True,
                 verbose=True):
        self.epochs = epochs
        self.verbose = verbose
        self.evaluate_steps = evaluate_steps
        self.valid_on_train_set = False
        self.valid_on_test_set = False

    def save_state(self, filepath, epoch, early_stopping):
        trainer_filepath = filepath + '.last_trainner_state.pkl'
        state = {
            'epocho': epoch,
            'earl_stopping': early_stopping
        }
        with open(trainer_filepath, 'wb') as fout:
            pickle.dump(state, fout)

    def load_state(self, filepath, epoch, early_stopping):
        state = {
            'epocho': epoch,
            'earl_stopping': early_stopping
        }
        trainer_filepath = filepath + '.last_trainner_state.pkl'
        try:
            with open(trainer_filepath, 'rb') as fin:
                state = pickle.load(fin)
        except:
            pass
        return state['epocho'], state['earl_stopping']

    def _train_epoch(self, model, train_loader, epoch):
        iter_data = (
            tqdm(
                train_loader,
                total=len(train_loader),
                ncols=100,
                desc=f"Train {epoch}:>5"
            )
            if self.verbose
            else train_loader
        )

        headers = None
        data = []
        for inputs, targets in iter_data:
            result = model.opt_one_batch(inputs, targets)
            assert isinstance(result, dict), "opt_one_batch must return a dict"
            assert 'loss' in result, "Returned dict must contain key 'loss'"
            if headers is None:
                headers = result.keys()
            data.append(list(result.values()))

        data = np.mean(np.array(data), axis=0)
        return dict(zip(headers, data))

    def _eval_data(self, model, dataloader, valid_fun):
        return model.eval_data(dataloader, valid_fun)

    def create_EarlyStopping(self, model):
        patience = 10
        delta = 0
        trace_func = print
        if hasattr(model, 'checkpoint_path'):
            checkpoint = model.checkpoint_path
        else:
            checkpoint = 'checkpoint'
        if hasattr(model, 'es_patience'):
            patience = model.es_patience
        if hasattr(model, 'es_delta'):
            delta = model.es_delta
        return EarlyStopping(patience, self.verbose, delta, trace_func, checkpoint)

    def train(self, model, train_loader, valid_loader, test_loader, valid_func, loger=sys.stdout):
        assert hasattr(model, 'opt_one_batch'), "Model must implement opt_one_batch"
        assert hasattr(model, 'eval_data'), "Model must implement eval_data"
        assert hasattr(model, 'save'), "Model must implement save"
        assert hasattr(model, 'load'), "Model must implement load"

        def printf(key, value, epoch):
            if isinstance(loger, SummaryWriter):
                loger.add_scalar(key, value, global_step=epoch)
            elif loger == sys.stdout:
                line = f"{key}={value}\t\t  epoch={epoch}"
                loger.write(line + "\n")

        metric_name = str(valid_func)
        early_stoping = self.create_EarlyStopping(model)
        epoch_start = 0

        for epoch in range(epoch_start + 1, self.epochs + 1):
            if early_stoping.early_stop:
                break

            results = self._train_epoch(model, train_loader, epoch)

            if self.verbose:
                for key, value in results.items():
                    printf(f"{key}/train", value, epoch)

            if self.evaluate_steps <= 0:
                continue

            if (epoch - 1) % self.evaluate_steps == 0:
                if valid_loader is not None:
                    val_score = self._eval_data(model, valid_loader, valid_func)
                    printf(f"{metric_name}/val", val_score, epoch)

            if valid_loader is not None and hasattr(valid_func, 'bigger') and valid_func.bigger is False:
                val_score = -val_score

            early_stoping(val_score, model, epoch)

        if self.evaluate_steps > 0:
            early_stoping.get_best(model)
