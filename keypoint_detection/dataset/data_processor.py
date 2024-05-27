import torch
from torch import Tensor


def postprocess_output(output: Tensor, img_size: tuple[int, int]):
    output = output.detach().cpu().numpy()

    ret = []
    for i in range(len(output)):
        xc = output[i, 0::2] * img_size[0]
        yc = output[i, 1::2] * img_size[1]

        converted_output = zip(xc, yc)
        ret.append(converted_output)

    return ret


def prepere_for_loss(target: Tensor):

    out = [item for t in target for item in t]

    return torch.stack(out, dim=-1).type(torch.float32)
