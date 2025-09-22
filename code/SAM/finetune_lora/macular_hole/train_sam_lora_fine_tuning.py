import os
import torch
import numpy as np
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import cv2
from torch.utils.data import Dataset, DataLoader
import loralib as lora
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import ndimage
from PIL import Image

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Paths
base_path = "/home/ig53agos/hole"
train_data_path = os.path.join(base_path, "Train_original_images")
train_mask_path = os.path.join(base_path, "Train_masks_images")
val_data_path = os.path.join(base_path, "Val_original_images")
val_mask_path = os.path.join(base_path, "Val_masks_images")
test_data_path = os.path.join(base_path, "Testing_original_images")
test_mask_path = os.path.join(base_path, "Testing_masks_images")

# Results folder
results_folder = os.path.join(base_path, "results", "sam_lora")
os.makedirs(results_folder, exist_ok=True)


img_size = 1024  

def preprocess_mask(mask_path):
    mask_img = Image.open(mask_path)
    mask_arr = np.array(mask_img)
    
    # Extract the red channel (macular hole regions)
    mask_red = (mask_arr[:, :, 0] == 255) & (mask_arr[:, :, 1] == 0) & (mask_arr[:, :, 2] == 0)
    mask_red = mask_red.astype(np.float32)
    
    # Improve mask quality
    mask_red = ndimage.binary_closing(mask_red, structure=np.ones((3,3)))
    mask_red = ndimage.binary_opening(mask_red, structure=np.ones((3,3)))
    mask_red = ndimage.gaussian_filter(mask_red.astype(float), sigma=0.5)
    mask_red = np.clip(mask_red, 0, 1)  # Ensure values are between 0 and 1
    
    return mask_red

def load_data(data_path, mask_path):
    images = []
    masks = []
    filenames = []
    transform = ResizeLongestSide(img_size)
    for filename in os.listdir(data_path):
        if filename.endswith(".png"):
            # Load and preprocess image
            img = cv2.imread(os.path.join(data_path, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform.apply_image(img)
            img = img.astype(np.float32) / 255.0
            images.append(img)
            
            # Load and preprocess mask
            mask = preprocess_mask(os.path.join(mask_path, filename))
            mask = transform.apply_image(mask)
            masks.append(mask)
            
            filenames.append(filename)
    
    return np.array(images), np.array(masks), filenames

class OCTDataset(Dataset):
    def __init__(self, images, masks, filenames):
        self.images = images
        self.masks = masks
        self.filenames = filenames

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).permute(2, 0, 1)
        mask = torch.from_numpy(self.masks[idx]).float().unsqueeze(0)
        
        
        if torch.any(mask < 0) or torch.any(mask > 1):
            print(f"Warning: Mask values out of range for file {self.filenames[idx]}")
            mask = torch.clamp(mask, 0, 1)
        
        return image, mask, self.filenames[idx]

def add_lora_to_sam(sam):
    for name, module in sam.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            lora.mark_only_lora_as_trainable(module)
    return sam

class SAMForTraining(torch.nn.Module):
    def __init__(self, sam_model):
        super().__init__()
        self.sam = sam_model
        self.mask_decoder = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 1, kernel_size=1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, image):
        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(image)
        
        masks = self.mask_decoder(image_embeddings)
        masks = torch.nn.functional.interpolate(masks, size=(image.shape[2], image.shape[3]), mode="bilinear", align_corners=False)
        return masks

def train(model, train_loader, val_loader, num_epochs, device, results_dir):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCELoss()

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = 0
        for images, masks, filenames in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            try:
                mask_predictions = model(images)
                loss = criterion(mask_predictions, masks)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            except RuntimeError as e:
                print(f"Error processing batch with files: {filenames}")
                print(f"Error message: {str(e)}")
                continue
        
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                mask_predictions = model(images)
                loss = criterion(mask_predictions, masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(results_dir, "sam_lora_best.pth"))

    return model

def iou(y_true, y_pred):
    intersection = torch.logical_and(y_true, y_pred).sum()
    union = torch.logical_or(y_true, y_pred).sum()
    return (intersection + 1e-6) / (union + 1e-6)

def dice_coef(y_true, y_pred):
    intersection = torch.logical_and(y_true, y_pred).sum()
    return (2. * intersection + 1e-6) / (y_true.sum() + y_pred.sum() + 1e-6)

def evaluate(model, test_loader, device):
    model.eval()
    total_iou = 0
    total_dice = 0
    num_samples = 0
    
    with torch.no_grad():
        for images, masks, _ in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            
            mask_predictions = model(images)
            pred_masks = (mask_predictions > 0.5).float()
            
            total_iou += iou(masks, pred_masks).item()
            total_dice += dice_coef(masks, pred_masks).item()
            num_samples += images.size(0)
    
    mean_iou = total_iou / num_samples
    mean_dice = total_dice / num_samples
    
    return mean_iou, mean_dice

def post_process_prediction(pred_mask, threshold=0.4):
    pred_mask = pred_mask.squeeze().cpu().numpy()
    binary_mask = (pred_mask > threshold).astype(np.float32)
    
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = ndimage.binary_opening(binary_mask, structure=kernel)
    binary_mask = ndimage.binary_closing(binary_mask, structure=kernel)
    
    return binary_mask

def visualize_results(model, test_loader, device, results_dir, num_samples=71):
    model.eval()
    
    with torch.no_grad():
        for i, (images, masks, filenames) in enumerate(tqdm(test_loader, desc="Visualizing")):
            if i >= num_samples:
                break
            
            images = images.to(device)
            masks = masks.to(device)
            
            mask_predictions = model(images)
            
            for j in range(images.size(0)):
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(images[j].cpu().permute(1, 2, 0))
                plt.title("Original Image")
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                gt_mask_display = np.zeros((*masks[j, 0].shape, 3))
                gt_mask_display[:, :, 0] = masks[j, 0].cpu().numpy()  # Red channel
                plt.imshow(gt_mask_display)
                plt.title("Ground Truth Mask")
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                pred_mask_processed = post_process_prediction(mask_predictions[j])
                pred_mask_display = np.zeros((*pred_mask_processed.shape, 3))
                pred_mask_display[:, :, 0] = pred_mask_processed  # Red channel
                plt.imshow(pred_mask_display)
                plt.title("Predicted Mask")
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f'result_{filenames[j]}'))
                plt.close()

    print(f"Visualization complete. Results saved in {results_dir}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(results_folder, f"sam_lora_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Load and preprocess data
    train_images, train_masks, train_filenames = load_data(train_data_path, train_mask_path)
    val_images, val_masks, val_filenames = load_data(val_data_path, val_mask_path)
    test_images, test_masks, test_filenames = load_data(test_data_path, test_mask_path)
    
    # Create datasets and dataloaders
    train_dataset = OCTDataset(train_images, train_masks, train_filenames)
    val_dataset = OCTDataset(val_images, val_masks, val_filenames)
    test_dataset = OCTDataset(test_images, test_masks, test_filenames)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)
    test_loader = DataLoader(test_dataset, batch_size=2)
    
    # Initialize and prepare the model
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = add_lora_to_sam(sam)
    sam_for_training = SAMForTraining(sam).to(device)
    
    # Train the model
    num_epochs = 50
    sam_for_training = train(sam_for_training, train_loader, val_loader, num_epochs, device, results_dir)
    
    # Load best model and evaluate
    sam_for_training.load_state_dict(torch.load(os.path.join(results_dir, "sam_lora_best.pth"), map_location=device))
    
    mean_iou, mean_dice = evaluate(sam_for_training, test_loader, device)
    print(f"Test Mean IoU: {mean_iou:.4f}")
    print(f"Test Mean Dice: {mean_dice:.4f}")
    
    with open(os.path.join(results_dir, "evaluation_results.txt"), "w") as f:
        f.write(f"Test Mean IoU: {mean_iou:.4f}\n")
        f.write(f"Test Mean Dice: {mean_dice:.4f}\n")
    
    # Visualize results
    visualize_results(sam_for_training, test_loader, device, results_dir)

    print(f"Results saved in {results_dir}")
    print("Done!")
