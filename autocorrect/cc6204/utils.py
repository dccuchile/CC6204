
import numpy as np
import torch


def check_list_types(a_list):
    def checker(value):
        if isinstance(value, list):
            for i, l in enumerate(value):
                value[i] = checker(l)

        else:
            if isinstance(value, (int, float)):
                pass
            elif isinstance(value, (np.ndarray, torch.Tensor)):
                value = value.tolist()
            else:
                raise ValueError(
                    'Supported submit values are numpy arrays, '
                    'torch tensors, python lists and int/floats. '
                    f'Answer type is: {type(value)}')

        return value

    return checker(a_list)
