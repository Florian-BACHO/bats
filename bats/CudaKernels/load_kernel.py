import pathlib
import cupy as cp

DIR_PATH = pathlib.Path(__file__).parent.resolve() / "CuFiles"

def load_kernel(filename, kernel_name, options=()):
    kernel_file_path = DIR_PATH / filename
    file = open(kernel_file_path, "r")
    file_content = file.read()
    file.close()
    return cp.RawKernel(file_content, kernel_name, options=options)