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

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class OCTDataset(Dataset):
    """
    Dataset class for OCT image segmentation handling AMD biomarkers
    """
    def __init__(self, data, image_size):
        self.data = data
        self.transform = ResizeLongestSide(image_size)
        # Biomarker color encoding matching thesis specifications
        self.biomarker_colors = [
            (0, 0, 0),      # Background
            (255, 0, 0),    # Drusen deposits
            (0, 255, 0),    # Retinal scarring
            (0, 0, 255)     # Fluid accumulation
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path = self.data[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path))
        
        image = self.transform.apply_image(image)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        mask_one_hot = np.zeros((len(self.biomarker_colors), mask.shape[0], mask.shape[1]), dtype=np.float32)
        for i, color in enumerate(self.biomarker_colors):
            mask_one_hot[i] = np.all(mask == color, axis=-1)
        
        transformed_mask = np.zeros((len(self.biomarker_colors), self.transform.target_length, self.transform.target_length), dtype=np.float32)
        for i in range(len(self.biomarker_colors)):
            transformed_mask[i] = cv2.resize(mask_one_hot[i], (self.transform.target_length, self.transform.target_length), interpolation=cv2.INTER_NEAREST)
        
        mask_one_hot = torch.from_numpy(transformed_mask).float()
        
        return image, mask_one_hot, str(img_path)

def add_lora_to_sam(sam):
   """
   Applies LoRA adaptation to SAM model layers
   """
   for name, module in sam.named_modules():
       if isinstance(module, torch.nn.Conv2d):
           lora.mark_only_lora_as_trainable(module)
   return sam

class SAMForTraining(torch.nn.Module):
   """
   SAM model adapted for OCT biomarker segmentation with LoRA
   """
   def __init__(self, sam_model):
       super().__init__()
       self.sam = sam_model
       self.mask_decoder = torch.nn.Sequential(
           torch.nn.Conv2d(256, 128, kernel_size=3, padding=1),
           torch.nn.BatchNorm2d(128),
           torch.nn.ReLU(),
           torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
           torch.nn.BatchNorm2d(64),
           torch.nn.ReLU(),
           torch.nn.Conv2d(64, 4, kernel_size=1)  # 4 classes: background + 3 biomarkers
       )
       
   def forward(self, image):
       with torch.no_grad():
           image_embeddings = self.sam.image_encoder(image)
       masks = self.mask_decoder(image_embeddings)
       masks = torch.nn.functional.interpolate(masks, size=(image.shape[2], image.shape[3]), mode="bilinear", align_corners=False)
       return masks

def train(model, train_loader, val_loader, num_epochs, device, results_dir):
   """
   Trains the SAM-LoRA model for OCT biomarker segmentation
   """
   model.train()
   optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
   criterion = torch.nn.BCEWithLogitsLoss()

   best_val_loss = float('inf')
   for epoch in range(num_epochs):
       train_loss = 0
       for images, masks, _ in tqdm(train_loader):
           images = images.to(device)
           masks = masks.to(device)
           
           optimizer.zero_grad()
           mask_predictions = model(images)
           loss = criterion(mask_predictions, masks)
           
           loss.backward()
           optimizer.step()
           train_loss += loss.item()
       
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

def evaluate(model, test_loader, device):
   """
   Evaluates model performance using IoU and Dice metrics for each biomarker
   """
   model.eval()
   total_iou = 0
   total_dice = 0
   num_samples = 0
   
   with torch.no_grad():
       for images, masks, _ in tqdm(test_loader, desc="Evaluating"):
           images = images.to(device)
           masks = masks.to(device)
           
           mask_predictions = model(images)
           pred_masks = (torch.sigmoid(mask_predictions) > 0.5).float()
           
           iou = calculate_iou(pred_masks, masks)
           dice = calculate_dice(pred_masks, masks)
           
           total_iou += iou.sum().item()
           total_dice += dice.sum().item()
           num_samples += images.size(0)
   
   mean_iou = total_iou / (num_samples * 4)  # 4 classes: background + 3 biomarkers
   mean_dice = total_dice / (num_samples * 4)
   
   return mean_iou, mean_dice

def calculate_iou(pred, target):
   """
   Calculates Intersection over Union
   """
   intersection = torch.logical_and(pred, target).sum((2, 3))
   union = torch.logical_or(pred, target).sum((2, 3))
   iou = (intersection.float() + 1e-6) / (union.float() + 1e-6)
   return iou

def calculate_dice(pred, target):
   """
   Calculates Dice coefficient
   """
   intersection = torch.logical_and(pred, target).sum((2, 3))
   return (2. * intersection.float() + 1e-6) / (pred.sum((2, 3)) + target.sum((2, 3)) + 1e-6)

def visualize_results(model, test_loader, device, results_dir):
   """
   Generates visualization of segmentation results with biomarker identification
   """
   model.eval()
   biomarker_colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
   biomarker_names = ['Background', 'Drusen', 'Scar', 'Liquid']
   
   with torch.no_grad():
       for i, (images, masks, img_paths) in enumerate(tqdm(test_loader, desc="Generating visualizations")):
           images = images.to(device)
           masks = masks.to(device)
           
           mask_predictions = model(images)
           pred_masks = (torch.sigmoid(mask_predictions) > 0.5).float()
           
           for j in range(images.size(0)):
               plt.figure(figsize=(20, 5))
               
               plt.subplot(1, 4, 1)
               plt.imshow(images[j].cpu().permute(1, 2, 0))
               plt.title("OCT B-scan")
               plt.axis('off')
               
               true_mask_rgb = np.zeros((masks.shape[2], masks.shape[3], 3), dtype=np.uint8)
               true_biomarkers = []
               for k in range(1, len(biomarker_colors)):  # Skip background
                   mask = masks[j, k].cpu().numpy() > 0.5
                   true_mask_rgb[mask] = biomarker_colors[k]
                   if np.any(mask):
                       true_biomarkers.append(biomarker_names[k])
               
               plt.subplot(1, 4, 2)
               plt.imshow(true_mask_rgb)
               plt.title(f"Ground Truth\n{', '.join(true_biomarkers)}")
               plt.axis('off')
               
               pred_mask_rgb = np.zeros((pred_masks.shape[2], pred_masks.shape[3], 3), dtype=np.uint8)
               pred_biomarkers = []
               for k in range(1, len(biomarker_colors)):  # Skip background
                   mask = pred_masks[j, k].cpu().numpy() > 0.5
                   pred_mask_rgb[mask] = biomarker_colors[k]
                   if np.any(mask):
                       pred_biomarkers.append(biomarker_names[k])
               
               plt.subplot(1, 4, 3)
               plt.imshow(pred_mask_rgb)
               plt.title(f"SAM-LoRA Prediction\n{', '.join(pred_biomarkers)}")
               plt.axis('off')
               
               plt.subplot(1, 4, 4)
               plt.text(0.1, 0.7, f"Ground Truth Biomarkers:\n{', '.join(true_biomarkers)}", fontsize=10, wrap=True)
               plt.text(0.1, 0.3, f"Predicted Biomarkers:\n{', '.join(pred_biomarkers)}", fontsize=10, wrap=True)
               plt.axis('off')
               
               plt.savefig(os.path.join(results_dir, f"result_{i}_{j}.png"))
               plt.close()

if __name__ == "__main__":
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   results_dir = f"results_sam_lora_{timestamp}"
   os.makedirs(results_dir, exist_ok=True)
   
   sam_checkpoint = "sam_vit_h_4b8939.pth"
   model_type = "vit_h"
   sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
   sam = add_lora_to_sam(sam)
   sam_for_training = SAMForTraining(sam).to(device)
   
   from prepare_dataset import load_dataset
   all_data = load_dataset("images", "masks", split=0)
   
   np.random.shuffle(all_data)
   train_split = int(0.7 * len(all_data))
   val_split = int(0.1 * len(all_data))
   train_data = all_data[:train_split]
   val_data = all_data[train_split:train_split+val_split]
   test_data = all_data[train_split+val_split:]
   
   image_size = 1024
   train_dataset = OCTDataset(train_data, image_size)
   val_dataset = OCTDataset(val_data, image_size)
   test_dataset = OCTDataset(test_data, image_size)
   
   train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=2)
   test_loader = DataLoader(test_dataset, batch_size=2)
   
   num_epochs = 50
   sam_for_training = train(sam_for_training, train_loader, val_loader, num_epochs, device, results_dir)
   
   sam_for_training.load_state_dict(torch.load(os.path.join(results_dir, "sam_lora_best.pth"), map_location=device))
   
   mean_iou, mean_dice = evaluate(sam_for_training, test_loader, device)
   print(f"Test Mean IoU: {mean_iou:.4f}")
   print(f"Test Mean Dice: {mean_dice:.4f}")
   
   with open(os.path.join(results_dir, "evaluation_results.txt"), "w") as f:
       f.write("SAM-LoRA Fine-tuning Results\n")
       f.write("===========================\n\n")
       f.write(f"Test Mean IoU: {mean_iou:.4f}\n")
       f.write(f"Test Mean Dice: {mean_dice:.4f}\n")
   
   visualize_results(sam_for_training, test_loader, device, results_dir)