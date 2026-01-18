import io
import numpy as np
import pickle
import sys

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from atp.trainer.earlystopping import EarlyStopping


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
        trainer_filepath = filepath + '.last_trainer_state.pkl'
        state = {
            'epoch': epoch,
            'early_stopping': early_stopping
        }
        with open(trainer_filepath, 'wb') as fout:
            pickle.dump(state, fout)

    def load_state(self, filepath, epoch, early_stopping):
        state = {
            'epoch': epoch,
            'early_stopping': early_stopping
        }
        trainer_filepath = filepath + '.last_trainer_state.pkl'
        try:
            with open(trainer_filepath, 'rb') as fin:
                state = pickle.load(fin)
        except:
            pass
        return state['epoch'], state['early_stopping']

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
        for batch_data in iter_data:
            result = model.opt_one_batch(batch_data)
            assert isinstance(result, dict), "opt_one_batch must return a dict"
            assert 'loss' in result, "opt_one_batch result must contain 'loss'"
            if headers is None:
                headers = result.keys()
            data.append(list(result.values()))
        data = np.mean(np.array(data), axis=0)
        return dict(zip(headers, data))

    def _eval_data(self, model, dataloader, valid_fun):
        return model.eval_data(dataloader, valid_fun)

    def create_EarlyStopping(self, model):
        patience = 7
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
        assert hasattr(model, 'opt_one_batch'), \
            f"Model {model} must implement opt_one_batch"
        assert hasattr(model, 'eval_data'), \
            f"Model {model} must implement eval_data"
        assert hasattr(model, 'save'), \
            f"Model {model} must implement save"
        assert hasattr(model, 'load'), \
            f"Model {model} must implement load"

        def printf(key, value, epoch):
            if isinstance(loger, SummaryWriter):
                loger.add_scalar(key, value, global_step=epoch)
            elif loger == sys.stdout:
                line = f"{key}={value}\t\t epoch={epoch}"
                loger.writelines(line)

        metric_name = str(valid_func)
        early_stopping = self.create_EarlyStopping(model)
        epoch_start = 0

        if hasattr(model, 'resume_state') and model.resume_state:
            epoch_start, early_stopping = self.load_state(
                model.checkpoint_path, epoch_start, early_stopping
            )
            model.load()

        for epoch in range(epoch_start + 1, self.epochs + 1):
            if early_stopping.early_stop:
                break

            results = self._train_epoch(model, train_loader, epoch)

            if self.verbose:
                for key, value in results.items():
                    printf(key, value, epoch)

            if self.evaluate_steps <= 0:
                continue

            if (epoch - 1) % self.evaluate_steps == 0:
                if self.valid_on_train_set:
                    train_score = self._eval_data(model, train_loader, valid_func)
                    printf(f"{metric_name}@train", train_score, epoch)

                if self.valid_on_test_set and test_loader is not None:
                    test_score = self._eval_data(model, test_loader, valid_func)
                    printf(f"{metric_name}@test", test_score, epoch)

                if valid_loader is not None:
                    val_score = self._eval_data(model, valid_loader, valid_func)
                    printf(f"{metric_name}@valid", val_score, epoch)
                elif self.valid_on_train_set:
                    val_score = train_score
                else:
                    val_score = -results['loss']

            if valid_loader is not None and hasattr(valid_func, 'bigger') and not valid_func.bigger:
                val_score = -val_score

            early_stopping(val_score, model, epoch)

            if hasattr(model, 'resume_state') and model.resume_state:
                self.save_state(model.checkpoint_path, epoch, early_stopping)
                model.save()

        if self.evaluate_steps > 0:
            early_stopping.get_best(model)
