import pycuda.driver as cuda
import pycuda.autoinit
import sys  # 导入sys模块来读取命令行参数

def get_cuda_device_major_minor(device_id=0):
    cuda.init()
    device = cuda.Device(device_id)
    attributes = device.get_attributes()
    major = attributes[cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR]
    minor = attributes[cuda.device_attribute.COMPUTE_CAPABILITY_MINOR]
    cmp_ver = f"{major}.{minor}"
    return cmp_ver

# 从命令行参数获取设备号
device_id = 0  # 默认设备号
if len(sys.argv) > 1:
    device_id = int(sys.argv[1])  # 将传入的参数转换为整数

cmp_ver = get_cuda_device_major_minor(device_id)
print(cmp_ver)  # 打印结果以便在Shell中捕获