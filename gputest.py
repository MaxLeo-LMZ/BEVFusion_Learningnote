import torch
import torchvision

torch_test1 = torch.__version__
torch_test2 =torchvision.__version__
print("torch version : ", torch_test1, "torch vision : ", torch_test2)

gpu_test1 = torch.cuda.is_available()
print(gpu_test1)
gpu_test2 =torch.cuda.device_count()
print(gpu_test2)
gpu_test3 =torch.cuda.current_device()
print(gpu_test3)
gpu_test4 =torch.cuda.get_device_name(0)
print(gpu_test4)
gpu_test5 = torch.version.cuda
print(gpu_test5)