"""
权重加载模块：把 HuggingFace safetensors 里的权重灌进我们自己的模型。

两个关键点：
1. 按"张量名"逐个加载，并且每个参数对象上都挂了一个 weight_loader 回调，
   由回调决定"这个参数在当前张量并行（TP）rank 上应该取哪一片"，
   于是同一份加载代码天然支持任意 TP 切分。
2. Qwen3 的 checkpoint 里 q_proj / k_proj / v_proj 是三个独立张量，
   而我们的模型把它们合并成了一个 qkv_proj。packed_modules_mapping
   就是描述"谁该被塞进合并参数里的哪一段"的映射表。
"""
import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """
    默认的权重加载器：不做任何切分，直接整体拷贝。

    适用于不需要 TP 切分的参数，例如 RMSNorm 的 weight。
    """
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """
    把 `path` 目录下所有 *.safetensors 权重文件加载进 `model`。

    一个模型可能被切成多个分片文件（model-00001-of-0000N.safetensors），
    所以先用 glob 把所有分片都找出来，逐个打开。
    """
    # 模型类上声明的"合并权重映射"，例如：
    #   {"q_proj": ("qkv_proj", "q"), "gate_proj": ("gate_up_proj", 0), ...}
    # 取不到就当成空字典（有些模型不需要这个机制）。
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        # safe_open(file, "pt", "cpu")：按 PyTorch 格式打开、先只映射到 CPU 内存（惰性加载）；
        # f.keys() 列出该分片里所有张量的名字。
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 先判断这个名字是否属于某个"被合并的模块"（例如权重名里含 q_proj）
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # v = 合并后在模型里的参数名（qkv_proj）；shard_id = 它是其中哪一段（"q"）
                        v, shard_id = packed_modules_mapping[k]
                        # 把 checkpoint 里的名字替换成我们模型里的名字
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        # 每个参数自带 loader：它知道自己是哪一段、该截取哪个 TP 分片
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        # 该权重已处理完，跳出内层循环
                        # （注意：跳出后 for...else 的 else 分支就不会执行）
                        break
                else:
                    # 没有命中任何合并规则：普通的一对一加载
                    param = model.get_parameter(weight_name)
                    # 参数上没挂 loader 就退回"整体拷贝"的默认实现
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
