import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import random
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Set random seeds for reproducibility
seed = None
if seed is not None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
else:
    torch.manual_seed(np.random.randint(0, 1000000))
    np.random.seed(np.random.randint(0, 1000000))
    random.seed(np.random.randint(0, 1000000))

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GaussianBumpDataset(Dataset):
    """
    Dataset of grayscale images with Gaussian bumps at different locations.

    Parameters
    ----------
    num_samples : int
        Number of samples in the dataset
    img_size : int
        Size of the square images (img_size x img_size)
    border : int
        Border width (in pixels) that the bump cannot be centered in
    bump_intensity : float
        Maximum intensity of the bumps
    noise_level : float
        Standard deviation of the Gaussian noise added to images
    border : int
        Border width (in pixels) that the bump cannot be centered in
    sigma_range : tuple[float, float]
        Range of standard deviations for the Gaussian bumps

    Attributes
    ----------
    images : torch.Tensor
        Tensor containing all generated images
    """

    def __init__(self, num_samples=1000, img_size=32, bump_intensity=1.0, noise_level=0.1, border=4, sigma_range=(0.9, 1.1)):
        self.num_samples = num_samples
        self.img_size = img_size
        self.bump_intensity = bump_intensity
        self.noise_level = noise_level
        self.border = border
        self.sigma_range = sigma_range

        # Generate the dataset
        self.images, self.bump_positions = self._generate_dataset()

    def _generate_gaussian_bump(self, center_x, center_y, sigma=2.0):
        """
        Generate a single Gaussian bump.

        Parameters
        ----------
        center_x : float
            x-coordinate of the bump center
        center_y : float
            y-coordinate of the bump center
        sigma : float
            Standard deviation of the Gaussian bump

        Returns
        -------
        np.ndarray
            2D array containing the Gaussian bump
        """
        x = np.arange(0, self.img_size, 1, float)
        y = x[:, np.newaxis]

        x0 = center_x
        y0 = center_y

        return self.bump_intensity * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))

    def _generate_dataset(self):
        """
        Generate the complete dataset of images with Gaussian bumps.

        Returns
        -------
        torch.Tensor
            Tensor of shape (num_samples, 1, img_size, img_size) containing all images
        """
        images = np.zeros((self.num_samples, 1, self.img_size, self.img_size))
        bump_positions = np.zeros((self.num_samples, 2))

        for i in range(self.num_samples):
            # Generate random positions for the bumps
            center_x = np.random.uniform(self.border, self.img_size - self.border)
            center_y = np.random.uniform(self.border, self.img_size - self.border)
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])

            # Add the bump to the image
            images[i, 0] += self._generate_gaussian_bump(center_x, center_y, sigma)
            bump_positions[i] = np.array([center_x, center_y])

            # Add Gaussian noise
            images[i, 0] += np.random.normal(0, self.noise_level, (self.img_size, self.img_size))

            # Clip values to [0, 1]
            images[i, 0] = np.clip(images[i, 0], 0, 1)

        return torch.FloatTensor(images), torch.FloatTensor(bump_positions)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample

        Returns
        -------
        torch.Tensor
            Image tensor of shape (1, img_size, img_size)
        """
        return self.images[idx], self.bump_positions[idx]


class SimCLRDataTransform:
    def __init__(self, max_translation=0.1, max_shear=30, noise_level=0.1):
        """
        Custom transform for noisy Gaussian bumps

        Args:
            max_translation (float): Maximum translation as a fraction of image size (0.1 = 10%)
            max_shear (float): Maximum shear angle in degrees
            noise_level (float): Standard deviation of the Gaussian noise to add
        """
        self.max_translation = max_translation
        self.max_shear = max_shear
        self.noise_level = noise_level

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # Convert tensor to correct shape if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # Add channel dimension if missing

        # Get image dimensions
        _, h, w = x.shape

        # Random translation values - small gentle movements
        tx = np.random.uniform(-self.max_translation, self.max_translation) * w
        ty = np.random.uniform(-self.max_translation, self.max_translation) * h

        # Random shear - gentle deformation
        shear_x = np.random.uniform(-self.max_shear, self.max_shear)
        shear_y = np.random.uniform(-self.max_shear, self.max_shear)

        # Create affine transformation matrix
        theta = torch.tensor([[1, shear_x / 100, tx / w], [shear_y / 100, 1, ty / h]], dtype=torch.float)

        # Reshape for grid_sample
        theta = theta.unsqueeze(0)

        # Create sampling grid
        grid = F.affine_grid(theta, [1, 1, h, w], align_corners=False)

        # Apply transformation
        x_transformed = F.grid_sample(x.unsqueeze(0), grid, align_corners=False).squeeze(0)

        # Add Gaussian noise
        noise = torch.randn_like(x_transformed) * self.noise_level
        x_transformed = x_transformed + noise

        # Ensure values are in valid range (0, 1)
        x_transformed = torch.clamp(x_transformed, 0, 1)

        return x_transformed

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(x), self.transform(x)


class SimCLREncoder(nn.Module):
    """
    Encoder network for SimCLR.

    This network consists of two convolutional layers followed by
    three fully connected layers.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    hidden_dim : int
        Dimension of the hidden layers
    output_dim : int
        Dimension of the output representation
    """

    def __init__(self, in_size=64, hidden_dim=8, output_dim=4):
        super(SimCLREncoder, self).__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(in_size * in_size, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim)
        """
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        return x


class SimCLR(nn.Module):
    """
    SimCLR model for contrastive learning.

    This model consists of an encoder network and a projection head.

    Parameters
    ----------
    encoder : nn.Module
        Encoder network
    projection_dim : int
        Dimension of the projection head output
    temperature : float
        Temperature parameter for the contrastive loss
    """

    def __init__(self, encoder: SimCLREncoder, projection_dim: int = 64, temperature: float = 0.5):
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.projection_head = nn.Sequential(
            nn.Linear(encoder.fc3.out_features, encoder.fc3.out_features), nn.ReLU(), nn.Linear(encoder.fc3.out_features, projection_dim)
        )
        self.temperature = temperature

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the SimCLR model.

        Parameters
        ----------
        x_i : torch.Tensor
            First set of augmented images
        x_j : torch.Tensor
            Second set of augmented images

        Returns
        -------
        tuple
            Tuple containing:
            - z_i: Projections of the first set of images
            - z_j: Projections of the second set of images
            - h_i: Representations of the first set of images
            - h_j: Representations of the second set of images
        """
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projection_head(h_i)
        z_j = self.projection_head(h_j)

        return z_i, z_j, h_i, h_j

    def contrastive_loss(self, z_i, z_j):
        """
        Compute the contrastive loss (NT-Xent loss).

        Parameters
        ----------
        z_i : torch.Tensor
            Projections of the first set of augmented images
        z_j : torch.Tensor
            Projections of the second set of augmented images

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        batch_size = z_i.shape[0]

        # Normalize the representations
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate the representations
        representations = torch.cat([z_i, z_j], dim=0)

        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        # Remove self-similarities
        mask = torch.eye(2 * batch_size, dtype=bool, device=device)
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)

        # Create labels for positive pairs (first offset due to removing self-similarities!)
        positives = torch.cat([torch.arange(batch_size, 2 * batch_size) - 1, torch.arange(batch_size)], dim=0).to(device)

        # Compute the NT-Xent loss
        logits = similarity_matrix / self.temperature
        loss = F.cross_entropy(logits, positives)

        return loss


def train_simclr(
    model: SimCLR,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int = 100,
    alpha: float = 1.0,
    beta: float = 1.0,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> list[float]:
    """
    Train the SimCLR model.

    Parameters
    ----------
    model : SimCLR
        SimCLR model to train
    data_loader : DataLoader
        DataLoader for the training data
    optimizer : torch.optim.Optimizer
        Optimizer for training
    epochs : int
        Number of training epochs
    alpha : float
        Weight for the contrastive component
    beta : float
        Weight for the repulsive component
    device : torch.device
        Device to run the training on

    Returns
    -------
    list
        List of training losses per epoch
    """
    model.train()
    losses = []

    progress_bar = tqdm(range(epochs), desc="Training the SimCLR model")
    for epoch in progress_bar:
        running_loss = 0.0

        for images, positions in data_loader:
            # Get the two augmented versions of each image
            x_i, x_j = images
            x_i, x_j = x_i.to(device), x_j.to(device)

            # Forward pass
            z_i, z_j, h_i, h_j = model(x_i, x_j)

            # Compute contrastive loss
            contrastive_component = model.contrastive_loss(z_i, z_j)

            # Compute repulsive component (optional)
            # This encourages representations to be more spread out
            z_all = torch.cat([z_i, z_j], dim=0)
            z_all_norm = F.normalize(z_all, dim=1)
            similarity = torch.mm(z_all_norm, z_all_norm.t())
            mask = torch.eye(z_all.shape[0], dtype=bool, device=device)
            repulsive_component = torch.mean(similarity[~mask])

            # Combine the components with weights
            loss = alpha * contrastive_component - beta * repulsive_component

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Record the average loss for this epoch
        epoch_loss = running_loss / len(data_loader)
        losses.append(epoch_loss)

        progress_bar.set_postfix(loss=epoch_loss)

    return losses


def visualize_representations(model, dataset, num_samples=200):
    """
    Visualize the learned representations using PCA.

    Parameters
    ----------
    model : SimCLR
        Trained SimCLR model
    dataset : Dataset
        Dataset to visualize
    num_samples : int
        Number of samples to visualize

    Returns
    -------
    None
    """
    model.eval()

    # Get a subset of the dataset
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    sample_images, sample_positions = zip(*[dataset[i] for i in indices])
    samples = torch.stack(sample_images)
    positions = torch.stack(sample_positions)

    # Get the representations
    with torch.no_grad():
        representations = model.encoder(samples.to(device)).cpu().numpy()

    pca = PCA(n_components=2)
    representations_2d = pca.fit_transform(representations)

    # Create a colormap based on positions
    # Normalize x and y positions to [0, 1] range
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    x_norm = (positions[:, 0] - x_min) / (x_max - x_min)
    y_norm = (positions[:, 1] - y_min) / (y_max - y_min)

    # Calculate radius (distance from center) and angle for each point
    # Assuming center is at (0.5, 0.5) after normalization
    center_x, center_y = 0.5, 0.5
    dx = x_norm - center_x
    dy = y_norm - center_y
    radius = np.sqrt(dx**2 + dy**2) / np.sqrt(0.5**2 + 0.5**2)  # Normalize by max possible radius
    angles = np.arctan2(dy, dx) / (2 * np.pi) + 0.5  # Convert [-π, π] to [0, 1]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")

    # Plot 1: Color by position in 2D grid using HSV color space
    # Hue represents angle (position around center), Saturation represents radius
    colors = [mcolors.hsv_to_rgb([angle, radius[i], 0.9]) for i, angle in enumerate(angles)]

    sc = ax1.scatter(representations_2d[:, 0], representations_2d[:, 1], c=colors, alpha=0.8, s=14)
    ax1.set_title("PCA of Representations")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Direct visualization of positions with same colors
    ax2.scatter(positions[:, 0], positions[:, 1], c=colors, alpha=0.8, s=14)
    ax2.set_title("2D Positions Samples")
    ax2.set_xlabel("X Position")
    ax2.set_ylabel("Y Position")
    ax2.grid(True, alpha=0.3)

    plt.show()


def visualize_dataset_samples(dataset, num_samples=5):
    """
    Visualize samples from the dataset.

    Parameters
    ----------
    dataset : Dataset
        Dataset to visualize
    num_samples : int
        Number of samples to visualize

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

    for i in range(num_samples):
        idx = np.random.randint(0, len(dataset))
        img = dataset[idx].squeeze().numpy()
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Sample {i+1}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show(block=True)


def visualize_augmentations(dataset, transform, num_samples=3):
    """
    Visualize the augmentations applied to samples.

    Parameters
    ----------
    dataset : Dataset
        Dataset to visualize
    transform : callable
        Transformation to apply
    num_samples : int
        Number of samples to visualize

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))

    for i in range(num_samples):
        idx = np.random.randint(0, len(dataset))
        original = dataset[idx].squeeze().numpy()

        # Apply the transformation to get two augmented versions
        aug1, aug2 = transform(dataset[idx])
        aug1 = aug1.squeeze().numpy()
        aug2 = aug2.squeeze().numpy()

        # Plot the original and augmented images
        axes[i, 0].imshow(original, cmap="gray")
        axes[i, 0].set_title(f"Original {i+1}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(aug1, cmap="gray")
        axes[i, 1].set_title(f"Augmentation 1")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(aug2, cmap="gray")
        axes[i, 2].set_title(f"Augmentation 2")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to run the SimCLR experiment.
    """
    # Create the dataset
    print("Creating dataset...")
    dataset = GaussianBumpDataset(num_samples=1000, img_size=32, bump_intensity=1.0, noise_level=0.1, border=4)

    # Visualize some samples from the dataset
    print("Visualizing dataset samples...")
    visualize_dataset_samples(dataset)

    # Create the data transformation
    transform = SimCLRDataTransform(max_translation=0.1, max_shear=30, noise_level=0.1)

    # Visualize the augmentations
    print("Visualizing augmentations...")
    visualize_augmentations(dataset, transform)

    # Create the data loader with proper handling of tensor inputs
    print("Creating data loader...")
    simclr_dataset = [(transform(img)) for img in dataset]
    batch_size = 128
    data_loader = DataLoader(simclr_dataset, batch_size=batch_size, shuffle=True)

    # Create the model
    hidden_dim = 32
    output_dim = 64
    projection_dim = 32

    encoder = SimCLREncoder(in_channels=1, hidden_dim=hidden_dim, output_dim=output_dim)
    model = SimCLR(encoder, projection_dim=projection_dim)
    model = model.to(device)

    # Create the optimizer
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    print("Training the model...")
    losses = train_simclr(model, data_loader, optimizer, epochs=50, alpha=1.0, beta=0.5)

    # Plot the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.show()

    # Visualize the learned representations
    print("Visualizing learned representations...")
    visualize_representations(model, dataset)


if __name__ == "__main__":
    main()
