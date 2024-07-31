import torch
import torch.optim as optim
import numpy as np
import time
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image
from skimage.transform import resize
from torchvision.models import inception_v3
from torchvision.transforms import functional as F
from torchvision import transforms
from generator import build_generator
from discriminator import build_discriminator
from scipy.linalg import sqrtm


# Preview image
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 16

# Size vector to generate images from
SEED_SIZE = 100

def hms_string(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}:{int(minutes)}:{int(seconds)}"

def calculate_fid(inception_model, real_images, generated_images):
    # Process images for FID calculation
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    real_images = torch.stack([transform(img) for img in real_images])
    generated_images = torch.stack([transform(img) for img in generated_images])
    
    inception_model.eval()
    with torch.no_grad():
        real_features = inception_model(real_images).cpu().numpy()
        generated_features = inception_model(generated_images).cpu().numpy()

    # Calculate FID score
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1.dot(sigma2))
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def visualize_feature_maps_by_index(generator, layer_index, noise_vector):
    # This function should visualize feature maps from a specific layer in the generator
    pass

def save_images(epoch, fixed_seed):
    # Convert seed to tensor and generate images
    fixed_seed = torch.tensor(fixed_seed, dtype=torch.float32)
    with torch.no_grad():
        generated_images = generator(fixed_seed).detach().cpu()
    save_image(generated_images, f"generated_images_epoch_{epoch + 1}.png", nrow=10)

def train(train_loader, epochs, generator, discriminator, criterion, optimizer_g, optimizer_d, inception_model):
    fixed_seed = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, SEED_SIZE))
    start = time.time()

    fid_scores = []
    generator_layer_index = 5
    noise_vector_for_visualization = np.random.normal(0, 1, (SEED_SIZE,))
    FID_INTERVAL = 10
    VISUALIZATION_INTERVAL = 500
    CHECKPOINT_PERIOD = 1000

    num_samples = 4000
    combined_images = np.random.normal(0, 1, (num_samples, SEED_SIZE))  # Dummy data

    print("Initializing training...")
    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}")
        epoch_start = time.time()
        gen_loss_list = []
        disc_loss_list = []

        for image_batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            image_batch = image_batch[0].to(device)  # Assuming image_batch is a tuple

            # Training Discriminator
            optimizer_d.zero_grad()
            noise = torch.randn(image_batch.size(0), SEED_SIZE, 1, 1, device=device)
            fake_images = generator(noise)
            real_output = discriminator(image_batch)
            fake_output = discriminator(fake_images.detach())
            disc_loss = (criterion(real_output, torch.ones_like(real_output)) +
                         criterion(fake_output, torch.zeros_like(fake_output))) / 2
            disc_loss.backward()
            optimizer_d.step()

            # Training Generator
            optimizer_g.zero_grad()
            fake_output = discriminator(fake_images)
            gen_loss = criterion(fake_output, torch.ones_like(fake_output))
            gen_loss.backward()
            optimizer_g.step()

            gen_loss_list.append(gen_loss.item())
            disc_loss_list.append(disc_loss.item())

        print(f'Epoch {epoch + 1}, gen loss={np.mean(gen_loss_list)}, disc loss={np.mean(disc_loss_list)}, {hms_string(time.time() - epoch_start)}')

        if (epoch + 1) % FID_INTERVAL == 0:
            noise = torch.randn(num_samples, SEED_SIZE, 1, 1, device=device)
            generated_images = generator(noise).detach().cpu()
            fid = calculate_fid(inception_model, combined_images, generated_images)
            print(f'Epoch {epoch + 1}, FID: {fid}')
            fid_scores.append(fid)

        if (epoch + 1) % VISUALIZATION_INTERVAL == 0:
            visualize_feature_maps_by_index(generator, generator_layer_index, noise_vector_for_visualization)

        if (epoch + 1) % CHECKPOINT_PERIOD == 0:
            torch.save(generator.state_dict(), f'generator_checkpoint_epoch_{epoch + 1}.pth')
            torch.save(discriminator.state_dict(), f'discriminator_checkpoint_epoch_{epoch + 1}.pth')
            print(f"Saved checkpoint for epoch {epoch + 1}")

        save_images(epoch, fixed_seed)

    print(f'Total Training time: {hms_string(time.time() - start)}')

    plt.figure(figsize=(10,5))
    plt.title("FID Scores over Epochs")
    plt.plot(fid_scores, label="FID Score")
    plt.xlabel("Epochs")
    plt.ylabel("FID Score")
    plt.legend()
    plt.show()

