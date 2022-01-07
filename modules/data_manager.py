import threading
import torch
import json
import os
from os import path
import shutil
import signal
import traceback
import subprocess
import requests



class Bunch(dict):
    def __init__(self, dictionary = None, **kwds):
        super().__init__(**kwds)
        self.__dict__ = self

        if dictionary != None:
            for key, val in dictionary.items():
                self[key] = val


class Logger:
    def __init__(self, do_log):
        self.do_log = do_log

    def __call__(self, msg):
        if self.do_log:
            print(msg)


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def run(command):
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output


class DataManager:

    def __init__(
        self,
        data={},
        get_models=lambda _:{},
        *,
        store_dir='train_data',
        server_url=None,
        save_parallel=True,
    ):
        self.save_parallel = save_parallel
        self.store_dir = store_dir if store_dir[-1] == '/' else (store_dir + '/')
        if not path.exists(self.store_dir):
            os.mkdir(self.store_dir)
        self.get_models = get_models
        self.use_cloud = True
        self.torchfiles_dir = path.join(store_dir, 'torchfiles')
        if not path.exists(self.torchfiles_dir):
            os.mkdir(self.torchfiles_dir)
        self.config_filename = path.join(self.store_dir, 'config.json')
        self.server_url = server_url
        self.done_saving = True
        self.to_ignore = list(self.__dict__.keys()) + ['to_ignore']
        self.dont_reload = []
        self.__dict__.update(data)
        self._load()

    def __str__(self):
        return str(self.__dict__)

    def _load(self):
        if path.exists('config.json'):
            with open('config.json') as f:
                config = json.load(f)
            self.__dict__.update(config)

        if (not path.exists(self.store_dir)) or len(os.listdir(self.store_dir)) == 0:
            print(f"{self.store_dir} is empty/non-existent.")
            if self.server_url != None:
                tmp_file = 'data.zip'
                if path.exists(tmp_file):
                    os.remove(tmp_file)
                run(f'wget {self.server_url} -O {tmp_file}')
                run(f'unzip {tmp_file} -d {self.store_dir}')
            # run(f'mv {self.store_dir}models/* {self.store_dir}
        config = {'dont_reload':[]}
        if path.exists(self.config_filename):
            with open(self.config_filename) as f:
                config = json.load(f)
            self.__dict__.update(config)
        pt_dict = self.get_models(self)
        self.__dict__.update(pt_dict)

        torchfiles = os.listdir(self.torchfiles_dir)
        for filename in torchfiles:
            splits = filename.split('.')
            prop = splits[0]
            if prop != '' and prop not in self.dont_reload:
                print('Loading:', prop)
                tensor = torch.load(path.join(self.torchfiles_dir, filename))
                if len(splits) > 2 and 'sd' == splits[-2]:
                    self.__dict__[prop].load_state_dict(tensor)
                else:
                    self.__dict__[prop] = tensor
                print('Reloaded: ', prop)

    def suicide(self):
        os.kill(os.getpid(), signal.SIGTERM)

    def is_json_serializable(self, obj):
        try:
            json.dumps(obj)
            return True
        except:
            return False

        def write_files(self, log):
            param_data = {}
            for key, val in self.__dict__.items():
                log('processing: ' + key)
                if key not in self.to_ignore:
                    if hasattr(val, 'state_dict') and callable(getattr(val, 'state_dict')):
                        new_filename = path.join(self.torchfiles_dir, key+'.sd.pt')
                        self.save_tensor(DataManager.to_cpu(val.state_dict()), new_filename)
                        log(f"Saved state dict: {new_filename}")
                    elif torch.is_tensor(val):
                        log(f"Saved tensor: {key}")
                        filename = path.join(self.torchfiles_dir, key+'.pt')
                        self.save_tensor(val.cpu(), filename)
                    elif self.is_json_serializable(val):
                        log("Saving param: "+key)
                        param_data[key] = val

            os.remove(self.config_filename)
            with open(self.config_filename, 'w') as f:
                json.dump(param_data, f, sort_keys=True, indent=4)
            try:
                with open(self.config_filename) as f:
                    json.load(f)
            except Exception as e:
                traceback.print_exc()
                print("Failed to write json cofig")
                self.suicide()
            log('Saved config file')

    def save_tensor(self, tensor, filename):
        if path.exists(filename):
            os.remove(filename)
        tries = 3
        success = False
        while not success:
            try:
                torch.save(tensor, filename);
            except Exception as e:
                print("Failed to save due to:\n",e)
                if tries == 0:
                    print("Couldn't save after 3 attempts, committing suicide.")
                    self.suicide()
                tries -= 1
            else:
                success = True

    def send_files(self, log):
        success = False
        i = 0
        while not success:
            if i > 2:
                print('Could not send models to server after 3 attempts. Exiting.')
                self.suicide()
            tmp_file = 'models.zip'
            if path.exists(tmp_file):
                os.remove(tmp_file)
            print('Zipping models dir')
            run(f'zip -r {tmp_file} {self.store_dir}')
            print('Done.')
            with open('models.zip', 'rb') as f:
                print('Uploading...')
                r = requests.post(f'http://{self.server_url}', files={'models.zip': f})
                print(r._content.decode('utf-8'))
                if r.status_code == 200:
                    success = True
            i += 1
        print('Done uploading!')


    def save(self, parallel=None, *, verbose=False):
        log = Logger(verbose)
        print('Started saving')
        def par_save():
            while not self.done_saving:
                pass
            self.done_saving = False
            self.write_files(log)
            if self.server_url != None:
                self.send_files(log)
            print("Saving complete.")
            self.done_saving = True

        if (parallel == None and self.save_parallel) or parallel:
            threading.Thread(target=par_save).start()
        else:
            par_save()

    @staticmethod
    def to_cpu(model_dict):
        cpu_model_dict = {}
        for key, val in model_dict.items():
            if torch.is_tensor(val):
                val = val.cpu()
            elif isinstance(val, dict):
                val = DataManager.to_cpu(val)
            cpu_model_dict[key] = val
        return cpu_model_dict
