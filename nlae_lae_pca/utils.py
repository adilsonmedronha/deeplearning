import matplotlib.pyplot as plt
import numpy as np
import torch



def visualize_batch(images, text="", num_images=4):
    images = images[:num_images]
    s = int(num_images ** 0.5)
    s_l, s_r = s, s
    if num_images == 4:
        s_l, s_r = 1, 4
    
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
    }
    
    images = images.reshape(-1, *images.shape[-2:])
    fig, axes = plt.subplots(s_l, s_r, figsize=(10, 10))
    str_ = "" if text == "" else f"Epoch {text}"
    plt.title(str_, fontdict=font)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_loss_history(infos, model_type):
    for key, value in infos.items():
        if key == "train_loss" or key == "test_loss":
            plt.plot(value, label=key)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_type}')
    plt.legend()
    plt.show()


def plot_latent_space(latent_space, model_type, labels):
    axis_x = latent_space[:, 0].tolist()
    axis_y = latent_space[:, 1].tolist()
    axis_z = latent_space[:, 2].tolist()
    fig = plt.figure(1, figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)
    plot = ax.scatter(axis_x, axis_y,axis_z,c=labels,cmap='Spectral',edgecolor=None,s=10)
    ax.set_title(f"Model {model_type}")
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    fig.colorbar(plot, boundaries=np.arange(10))
    plt.show()


def visualize_reconstruction(original, out, labels, model_type):
    n = 8
    plt.figure(figsize=(20, 5))
    plt.suptitle(f"{model_type} Reconstruction", fontsize=16)
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.title(int(labels[i]))
        plt.imshow(original[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        r = out[i].reshape(28, 28) if model_type == "PCA" else out[i].reshape(28, 28)

        plt.imshow(r)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
    plt.show()


def separate_by_class(data, labels):
    separated = {class_:[] for class_ in range(10)}
    for d, l in zip(data, labels):
        separated[int(l)].append(d)
    for i in range(10):
        separated[i] = np.array(separated[i])
    return separated


def euclidean_distance(x, y):
    assert x.shape == y.shape
    return torch.sqrt(((x - y) ** 2).sum()).item()