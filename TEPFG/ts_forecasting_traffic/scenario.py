import logging
import os
import shutil
import time
import hashlib


def myhash(data):
    return hashlib.md5(data.encode(encoding='UTF-8')).hexdigest()


def check_result_existing(log_file):
    if os.path.isfile(log_file):
        return True
    return False


class ScenarioAbstract:

    def run(self, algorithm, protocol, metrics, data_path, out_path, over_write):

        data_name = os.path.splitext(data_path)[0]
        data_name = os.path.split(data_name)[-1]

        cache_dir = os.path.join(
            out_path,
            algorithm.class_name() + '-' + data_name,
            'cache'
        )

        out_path = os.path.join(
            out_path,
            algorithm.class_name() + '-' + data_name,
            str(myhash(str(algorithm)))
        )

        if os.path.isdir(out_path):
            if over_write:
                shutil.rmtree(out_path)
                os.makedirs(out_path, exist_ok=True)
            else:
                if hasattr(algorithm, 'resume_state') and algorithm.resume_state:
                    os.makedirs(out_path, exist_ok=True)
                else:
                    print(f"The result directory already exists{out_path}")
                    exit(-1)
        else:
            os.makedirs(out_path)

        log_path = os.path.join(out_path, 'result.log')
        start_time = str(time.strftime("%Y%m%d%H%M%S", time.localtime()))

        algorithm.checkpoint_path = os.path.join('checkpoint', out_path, 'checkpoint')
        os.makedirs(os.path.join('checkpoint', out_path), exist_ok=True)

        if not hasattr(algorithm, "tensorboard_path") or not algorithm.tensorboard_path:
            algorithm.tensorboard_path = os.path.join(out_path, 'log_tensorboard')

        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        if not hasattr(algorithm, 'cache_dir') or not algorithm.cache_dir:
            algorithm.cache_dir = cache_dir

        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format="%(message)s"
        )

        logging.info('Train Start Time: ' + start_time)
        logging.info("ds" + ":" + data_path)
        logging.info("algorithm-parameter:")
        logging.info(algorithm)

        results = protocol.test(algorithm, metrics)

        logging.info("RESULTS=" + str(results))
        logging.info('Train Stop Time: ' + str(time.strftime("%Y%m%d%H%M%S", time.localtime())))

        assert len(results) == len(metrics)

        return results

    def parse_data(self, data_dir):
        raise Exception("This is an Abstract Class, Do Not Call It")
