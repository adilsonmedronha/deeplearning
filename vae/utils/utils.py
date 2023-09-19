import matplotlib.pyplot as plt
import torchvision
import torch
import imageio
import os
from PIL import Image
from tqdm import tqdm
import argparse
from models import Lvae, PCvae, FCvae

def viz(images_batch, save_path, plot_in_jupyter=False):
    grid = torchvision.utils.make_grid(images_batch, nrow=8, padding=2)
    if torch.is_tensor(grid):  
        grid = grid.detach().cpu().numpy() 
    grid = grid.transpose((1, 2, 0))

    if plot_in_jupyter:
        plt.figure(figsize=(40, 40))
        plt.imshow(grid)
        plt.axis('off')
        plt.show()
    else:
        plt.imsave(save_path, grid)
        plt.close()
        

def create_gif_from_folder(input_folder, output_gif_path, duration=0.2):
    images = []
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))])
    print("gif generation")
    for filename in tqdm(image_files):
        image_path = os.path.join(input_folder, filename)
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(img)
    imageio.mimsave(output_gif_path, images, duration=duration)


def plot_losses(train_losses, val_losses, save_path):
    plt.figure()
    epochs = len(train_losses)
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def save_model(model, save_path, name):
    if not os.path.exists(save_path): os.makedirs(save_path)
    torch.save(model.state_dict(),f"{save_path}/{name}")

def get_models(args, device):
    model_classes = {
        "lvae": Lvae,
        "pcvae": PCvae,
        "fcvae": FCvae,
    }

    if args.model_type in model_classes:
        selected_model_class = model_classes[args.model_type]
        return selected_model_class(args.z_dim, 
                                    args.img_shape, 
                                    args.w_init_method, device)
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")


def get_args():
    parser = argparse.ArgumentParser(description='A simple program with arguments')
    parser.add_argument('--model_type', type=str, help='')
    parser.add_argument('--run_name', type=str, help='')
    parser.add_argument('--dataset_path', type=str, help='')
    parser.add_argument('--dataset_name', type=str, help='MNIST or CelebA')
    parser.add_argument('--img_shape', type=int, nargs='+', help='hw')  # Example: --img_shape 3 64 64
    parser.add_argument('--batch_size', type=int, help='hw')
    parser.add_argument('--n_images', type=int, help='hw')
    parser.add_argument('--epochs', type=int, help='An integer argument')
    parser.add_argument('--z_dim', type=int, help='latent dimension')
    parser.add_argument('--early_stop_patience', type=int, default=3, help='A float argument')
    parser.add_argument('--lr', type=float, default=1e-6, help='A float argument')
    parser.add_argument('--w_init_method', type=str, help='')
    parser.add_argument('--model_save_path_best_loss_train', type=str, help='A string argument')
    parser.add_argument('--model_save_path_best_loss_val', type=str, help='A string argument')
    parser.add_argument('--model_save_path_last', type=str, help='A string argument')
    parser.add_argument('--path2results', type=str, help='A string argument')  # "./results/celeba/during_training/"
    parser.add_argument('--path2gif', type=str, help='A string argument')
    args = parser.parse_args()
    print("Arguments:")
    for key, value in vars(args).items():
        print(f"\t{key}: {value}")
    return args