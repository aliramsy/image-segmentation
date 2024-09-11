import matplotlib.pyplot as plt
import numpy
import torch


def plot_image(tensor):
    if tensor.shape[0] == 3:
        image = tensor.permute(1, 2, 0).cpu().numpy()
        plt.imshow(image)
    elif tensor.shape[0] == 1:
        image = tensor.squeeze().cpu().numpy()
        plt.imshow(image, cmap='gray')
    else:
        raise ValueError(f"Unexpected number of channels: {tensor.shape[0]}")

    plt.axis('off')
    plt.show()


def display_image(output):
    # output shape is (1, height, width)
    # mask shape is (1, height, width)
    mask = torch.zeros((1, output.shape[1], output.shape[2]))
    value_list = [.1, .25, .4, .55, .7, .85]
    for i, j in zip(range(6), value_list):
        mask[0][output[0] == i] = j

    return mask
