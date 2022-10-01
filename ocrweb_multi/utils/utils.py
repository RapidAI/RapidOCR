import json
import time
import warnings

from onnxruntime import (
    get_available_providers,
    get_device,
    SessionOptions,
    InferenceSession,
)
from utils.config import conf, get_resource_path


def parse_bool(val):
    if not isinstance(val, str):
        return bool(val)
    return val.lower() in ('1', 'true', 'yes')


def default(obj):
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    return obj


def tojson(obj, **kws):
    return json.dumps(obj, default=default, ensure_ascii=False, **kws) + '\n'


class OrtInferSession(object):
    def __init__(self, model_path):
        ort_conf = conf['global']
        sess_opt = SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False

        cuda_ep = 'CUDAExecutionProvider'
        cpu_ep = 'CPUExecutionProvider'

        providers = []
        if (
            ort_conf['use_cuda']
            and get_device() == 'GPU'
            and cuda_ep in get_available_providers()
        ):
            providers = [(cuda_ep, ort_conf[cuda_ep])]

        providers.append(cpu_ep)

        self.session = InferenceSession(
            str(get_resource_path(model_path)),
            sess_options=sess_opt,
            providers=providers,
        )

        if ort_conf['use_cuda'] and cuda_ep not in self.session.get_providers():
            warnings.warn(
                f'{cuda_ep} is not avaiable for current env, the inference part is automatically shifted to be executed under {cpu_ep}.\n'
                'Please ensure the installed onnxruntime-gpu version matches your cuda and cudnn version, '
                'you can check their relations from the offical web site: '
                'https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html',
                RuntimeWarning,
            )

    def get_input_name(self, input_idx=0):
        return self.session.get_inputs()[input_idx].name

    def get_output_name(self, output_idx=0):
        return self.session.get_outputs()[output_idx].name


class Ticker(object):
    def __init__(self, reset=True) -> None:
        self.ts = time.perf_counter()
        self.reset = reset
        self.maps = {}

    def tick(self, name, reset=None):
        ts = time.perf_counter()
        if reset is None:
            reset = self.reset
        dt = ts - self.ts
        if reset:
            self.ts = ts
        self.maps[name] = dt
        return dt
