# STG-NF modules, based on awesome previous work by https://github.com/y0ast/Glow-PyTorch


def compute_same_pad(kernel_size, stride):
    # Ensure kernel_size and stride are lists or tuples.
    kernel_size = (
        kernel_size if isinstance(kernel_size, (list, tuple)) else [kernel_size]
    )
    stride = stride if isinstance(stride, (list, tuple)) else [stride]
    assert len(kernel_size) == len(stride), (
        "Pass kernel size and stride both as int or as equal-length iterables"
    )
    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]


# def split_feature(tensor, type="split", imgs=False):
#     C = tensor.size(1)
#     if type == "split":
#         a = tensor[:, : C // 2, ...]
#         b = tensor[:, C // 2 :, ...]
#     elif type == "cross":
#         a = tensor[:, 0::2, ...]
#         b = tensor[:, 1::2, ...]
#     else:
#         raise ValueError("Invalid type, choose 'split' or 'cross'")
#     if not imgs:
#         a = a.squeeze(dim=1)
#         b = b.squeeze(dim=1)
#     return a, b


def split_feature(tensor, type="split", imgs=False):
    C = tensor.size(1)
    if type == "split":
        return tensor[:, : C // 2, ...], tensor[:, C // 2 :, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]
    else:
        raise ValueError("Invalid type, choose 'split' or 'cross'")
