import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os
import time
from tqdm import tqdm
from scipy.linalg import sqrtm
from visualization_utils import visualize_feature_maps_by_index
from torchvision.models import inception_v3, Inception_V3_Weights


# Assuming build_generator and build_discriminator are defined similarly in PyTorch
from generator import build_generator
from discriminator import build_discriminator

# Generation resolution - Must be square
GENERATE_RES = 5  # Generation resolution factor (1=32, 2=64, 3=96, etc.)
GENERATE_SQUARE = 32 * GENERATE_RES  # rows/cols (should be square)
IMAGE_CHANNELS = 3

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

# Preview image
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 16

# Size vector to generate images from
SEED_SIZE = 100

# Configuration
DATA_PATH = './data'  # Path to your local dataset
EPOCHS = 250
CHECKPOINT_DIR = './training_checkpoints'
CHECKPOINT_PERIOD = 70
BATCH_SIZE = 64
BUFFER_SIZE = 60000

print(f"Will generate {GENERATE_SQUARE}px square images.")

# Directory where augmented images are saved
augmented_data_dir = './augmented_data'

# Load Images Function
class ImageDataset(Dataset):
    def __init__(self, directory, image_size, transform=None):
        self.directory = directory
        self.image_size = image_size
        self.transform = transform
        self.images = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).resize(self.image_size)
        if self.transform:
            img = self.transform(img)
        return img

transform = transforms.Compose([
    transforms.Resize((GENERATE_SQUARE, GENERATE_SQUARE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * IMAGE_CHANNELS, [0.5] * IMAGE_CHANNELS)
])

original_dataset = ImageDataset(os.path.join(DATA_PATH, 'images'), (GENERATE_SQUARE, GENERATE_SQUARE), transform)
augmented_dataset = ImageDataset(augmented_data_dir, (GENERATE_SQUARE, GENERATE_SQUARE), transform)

combined_dataset = original_dataset + augmented_dataset
train_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("All images loaded.")
print(f"Total number of images loaded: {len(combined_dataset)}")

# Initialize the GAN components with error handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = build_generator(SEED_SIZE, IMAGE_CHANNELS).to(device)
discriminator = build_discriminator((GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS)).to(device)

# Optimizers
generator_optimizer = optim.AdamW(generator.parameters(), lr=1.7e-4, betas=(0.5, 0.999), eps=1e-07, weight_decay=0.001)
discriminator_optimizer = optim.AdamW(discriminator.parameters(), lr=1.5e-4, betas=(0.5, 0.999), eps=1e-07, weight_decay=0.001)

# Loss function
criterion = nn.BCEWithLogitsLoss()

# Load InceptionV3 model with the correct weights parameter
inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device)
inception_model.eval()

def calculate_fid(model, real_images, generated_images):
    def preprocess(images):
        images = (images * 127.5 + 127.5) / 255.0
        images = np.array([np.array(transforms.Resize((299, 299))(Image.fromarray((img * 255).astype(np.uint8)))) for img in images])
        images = torch.tensor(images).permute(0, 3, 1, 2).float()
        return images

    real_images = preprocess(real_images)
    generated_images = preprocess(generated_images)

    def get_activations(images):
        with torch.no_grad():
            pred = model(images.to(device))
        return pred.cpu().numpy()

    act1 = get_activations(real_images)
    act2 = get_activations(generated_images)

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

smooth_factor = 0.9

def train_step(images):
    batch_size = images.size(0)
    real_labels = torch.ones(batch_size, 1, device=device)  # Adjust shape to [batch_size, 1]
    fake_labels = torch.zeros(batch_size, 1, device=device)  # Adjust shape to [batch_size, 1]

    noise = torch.randn(batch_size, SEED_SIZE, device=device)

    discriminator_optimizer.zero_grad()

    real_output = discriminator(images)
    fake_images = generator(noise)
    fake_output = discriminator(fake_images.detach())

    real_loss = criterion(real_output, real_labels)
    fake_loss = criterion(fake_output, fake_labels)
    disc_loss = real_loss + fake_loss

    disc_loss.backward()
    discriminator_optimizer.step()

    generator_optimizer.zero_grad()

    output = discriminator(fake_images)
    gen_loss = criterion(output, real_labels)

    gen_loss.backward()
    generator_optimizer.step()

    return gen_loss.item(), disc_loss.item()

def save_images(cnt, noise):
    generator.eval()
    with torch.no_grad():
        fake_images = generator(noise).detach().cpu()
    generator.train()

    image_grid = np.full((PREVIEW_MARGIN + (PREVIEW_ROWS * (GENERATE_SQUARE + PREVIEW_MARGIN)),
                          PREVIEW_MARGIN + (PREVIEW_COLS * (GENERATE_SQUARE + PREVIEW_MARGIN)), IMAGE_CHANNELS),
                         255, dtype=np.uint8)

    for i in range(PREVIEW_ROWS):
        for j in range(PREVIEW_COLS):
            image = (fake_images[i * PREVIEW_COLS + j].permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8)
            image_grid[PREVIEW_MARGIN + i * (GENERATE_SQUARE + PREVIEW_MARGIN): PREVIEW_MARGIN + i * (GENERATE_SQUARE + PREVIEW_MARGIN) + GENERATE_SQUARE,
                       PREVIEW_MARGIN + j * (GENERATE_SQUARE + PREVIEW_MARGIN): PREVIEW_MARGIN + j * (GENERATE_SQUARE + PREVIEW_MARGIN) + GENERATE_SQUARE, :] = image

    output_path = os.path.join('output', f"trained-{cnt}.png")
    os.makedirs('output', exist_ok=True)
    Image.fromarray(image_grid).save(output_path)

# Train the GAN
fixed_noise = torch.randn(PREVIEW_ROWS * PREVIEW_COLS, SEED_SIZE, device=device)

start = time.time()
for epoch in range(EPOCHS):
    gen_loss_list = []
    disc_loss_list = []

    for images in train_loader:
        images = images.to(device)
        gen_loss, disc_loss = train_step(images)
        gen_loss_list.append(gen_loss)
        disc_loss_list.append(disc_loss)

    avg_gen_loss = np.mean(gen_loss_list)
    avg_disc_loss = np.mean(disc_loss_list)

    if (epoch + 1) % CHECKPOINT_PERIOD == 0:
        torch.save({
            'epoch': epoch + 1,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'generator_optimizer_state_dict': generator_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
        }, os.path.join(CHECKPOINT_DIR, f'checkpoint_{epoch+1}.pth'))

    print(f'Epoch {epoch+1}, Gen Loss: {avg_gen_loss}, Disc Loss: {avg_disc_loss}')

    if (epoch + 1) % CHECKPOINT_PERIOD == 0:
        save_images(epoch + 1, fixed_noise)

    # Evaluate FID every 50 epochs
    if (epoch + 1) % 50 == 0:
        noise = torch.randn(len(original_dataset), SEED_SIZE, device=device)
        with torch.no_grad():
            generated_images = generator(noise).detach().cpu().permute(0, 2, 3, 1).numpy()
        fid_score = calculate_fid(inception_model, original_dataset[:len(generated_images)], generated_images)
        print(f"FID score at epoch {epoch + 1}: {fid_score}")

# Save final models
torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, 'generator_final.pth'))
torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_DIR, 'discriminator_final.pth'))

end = time.time()
print(f'Training finished in {hms_string(end - start)}')
