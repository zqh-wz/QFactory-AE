import os
import re
import uuid
import pickle
import hashlib
import logging
import subprocess
from torch.utils.cpp_extension import CUDA_HOME

from .runtime import Runtime, RuntimeCache

logger = logging.getLogger(__name__)

runtime_cache = RuntimeCache()

def hash_to_hex(s: str) -> str:
    md5 = hashlib.md5()
    md5.update(s.encode('utf-8'))
    return md5.hexdigest()[0:12]

def get_qfactory_version() -> str:
    include_dir = f'{os.path.dirname(os.path.abspath(__file__))}/../include'
    md5 = hashlib.md5()
    for root, _, files in os.walk(include_dir):
        for filename in filter(lambda x: x.endswith('.h'), sorted(files)):
            with open(os.path.join(root, filename), 'rb') as f:
                md5.update(f.read())
    return md5.hexdigest()[0:12]

def get_nvcc_compiler():
    nvcc_path = f'{CUDA_HOME}/bin/nvcc'
    version_pattern = re.compile(r'release (\d+\.\d+)')
    try:
        match = version_pattern.search(os.popen(f'{nvcc_path} --version').read())
        version = match.group(1)
        assert match is not None
    except:
        raise RuntimeError(f'Cannot get the version of NVCC compiler, CUDA_HOME={CUDA_HOME}')
    return nvcc_path, version

def get_cache_dir():
    if 'QFACTORY_CACHE_DIR' in os.environ:
        path = os.getenv('QFACTORY_CACHE_DIR')
        os.makedirs(path, exist_ok=True)
        return path
    return os.path.expanduser('~') + '/.qfactory'

def write_file(tmp_dir: str, path: str, content: str):
    # Atomic write
    tmp_file_path = f'{tmp_dir}/file.tmp.{str(uuid.uuid4())}.{hash_to_hex(path)}'
    with open(tmp_file_path, 'w') as f:
        f.write(content)
    os.replace(tmp_file_path, path)


class JITCompiler:
    def __init__(self):
        pass

    def compile(
        self,
        qfunc,
    ) -> Runtime:
        code = qfunc.emit()
        args_name_type = qfunc.arg_name_type()

        arch = os.getenv('QFACTORY_ARCH')
        if arch is None:
            logger.error('QFACTORY_ARCH is not set, please set it to the target architecture')
            raise RuntimeError('QFACTORY_ARCH is not set')

        nvcc_flags = [
            '-std=c++20', '-shared', '-O3', '--expt-relaxed-constexpr', '--resource-usage', '-Xptxas=-warn-lmem-usage',
            f'-gencode=arch=compute_{arch},code=sm_{arch}'
        ]
        cxx_flags = ['-fPIC', '-O3']
        flags = [*nvcc_flags, f'--compiler-options={",".join(cxx_flags)}']
        include_dirs = [
            f'{os.path.dirname(os.path.abspath(__file__))}/../include',
        ]

        signature = f'{code}$${get_nvcc_compiler()}$${get_qfactory_version()}$${flags}$${arch}'
        runtime_name = f'kernel.{arch}.{hash_to_hex(signature)}'
        runtime_path = f'{get_cache_dir()}/runtimes/{runtime_name}'
        tmp_dir = f'{get_cache_dir()}/tmp'
        
        global runtime_cache
        if runtime_cache[runtime_path] is not None:
            logger.debug(f"Using cached runtime for {runtime_name}")
            return runtime_cache[runtime_path]
        
        os.makedirs(runtime_path, exist_ok=True)
        os.makedirs(tmp_dir, exist_ok=True)
        args_path = f'{runtime_path}/kernel.args'
        src_path = f'{runtime_path}/kernel.cu'
        so_path = f'{runtime_path}/kernel.so'
        tmp_so_path = f'{tmp_dir}/nvcc.tmp.{str(uuid.uuid4())}.{hash_to_hex(so_path)}.so'

        write_file(tmp_dir, src_path, code)

        command = [get_nvcc_compiler()[0],
                src_path, '-o', tmp_so_path,
                *flags,
                *[f'-I{d}' for d in include_dirs]]
        logger.info(f'Compiling kernel with command: {command}')
        return_code = subprocess.check_call(command)
        assert return_code == 0, f'Failed to compile {src_path}'

        pickle.dump(args_name_type, open(args_path, 'wb'))
        os.replace(tmp_so_path, so_path) # Atomic write

        runtime_cache[runtime_path] = Runtime(runtime_path)
        return runtime_cache[runtime_path]


jit = JITCompiler()