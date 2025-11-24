import torch
from thop import profile, clever_format
from lib.full_LGUNet_newest_02 import Network

# 导入你的模型类

# 创建模型实例
model = Network(channel=64,imagenet_pretrained=False)
model.eval()  # 设置为推理模式

# 设置输入张量的大小（根据你模型实际需求修改）
dummy_input = torch.randn(1, 3, 512, 512)  # COD常见输入尺寸

# 计算 FLOPs 和参数量
flops, params = profile(model, inputs=(dummy_input,))
flops, params = clever_format([flops, params], "%.2f")

print(f"Params: {params}")
print(f"FLOPs: {flops}")
