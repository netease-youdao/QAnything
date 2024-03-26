gpu_type = ''


def set_gpu_type(type):
    global gpu_type
    gpu_type = type.lower()


def get_gpu_type():
    global gpu_type
    return gpu_type
