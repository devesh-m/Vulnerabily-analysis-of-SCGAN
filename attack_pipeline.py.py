import os
import torch
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from easydict import EasyDict as edict
from data_test import get_loader
from torch.autograd import Variable
from model import G_RLS
import cv2
import time
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1  # The Biometric Judge

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.sys.path.append(BASE_DIR)
os.chdir(BASE_DIR)

def get_target_embedding(facenet, image_path, device):
    """Loads the target identity we want the GAN to spoof."""
    img = cv2.imread(image_path)
    if img is None:
        print("WARNING: target_identity.jpg not found. Using a random target embedding for testing.")
        return torch.randn(1, 512).to(device)
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 160))
    img = np.transpose(img, (2, 0, 1)) # HWC to CHW
    img = torch.from_numpy(img).float().unsqueeze(0).to(device) / 255.0
    img = (img - 0.5) / 0.5 # Standard normalization for facenet
    
    with torch.no_grad():
        target_emb = facenet(img)
    return target_emb

def apply_median_defense(tensor_img):
    """Applies a 3x3 Median Filter as a zero-shot purification defense."""
    # Move tensor to CPU and convert to HWC numpy array
    img_np = tensor_img.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0)
    
    # Normalize to 0-255 uint8 for OpenCV processing
    v_min, v_max = img_np.min(), img_np.max()
    img_norm = (img_np - v_min) / (v_max - v_min + 1e-8)
    img_uint8 = (img_norm * 255).astype(np.uint8)
    
    # Execute the defensive spatial filter (destroys high-frequency PGD noise)
    blurred = cv2.medianBlur(img_uint8, 3)
    
    # Rescale back to original tensor distribution space
    blurred_float = blurred.astype(np.float32) / 255.0
    blurred_rescaled = blurred_float * (v_max - v_min) + v_min
    
    # Return to GPU as a PyTorch tensor
    return torch.from_numpy(blurred_rescaled.transpose(2, 0, 1)).unsqueeze(0).to(tensor_img.device)

def main():
    torch.manual_seed(1)
    np.random.seed(0)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    
    opt = edict()
    opt.nGPU = 1
    opt.batchsize = 1
    opt.cuda = torch.cuda.is_available()
    device = torch.device('cuda' if opt.cuda else 'cpu')
    cudnn.benchmark = True
    print('device:', device)
    
    print('========================LOAD MODELS============================')
    # 1. Load SCGAN (The Target)
    net_G_RLS = G_RLS().to(device)
    a = torch.load('./model/pretrained_model.pth', map_location=device)["G_l2h"]
    net_G_RLS.load_state_dict(a)
    net_G_RLS.eval()
    for param in net_G_RLS.parameters():
        param.requires_grad = False # Freeze SCGAN completely
        
    # 2. Load FaceNet (The Judge)
    print("Loading FaceNet Model...")
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    for param in facenet.parameters():
        param.requires_grad = False # Freeze FaceNet completely
        
    # 3. Get the embedding of the person we want to spoof
    target_emb = get_target_embedding(facenet, './target_identity.jpg', device)

    data_name = 'FFHQ'
    test_loader = get_loader(data_name, opt.batchsize)
    test_save = './attack_results'
    if not os.path.exists(test_save):
        os.makedirs(test_save)

    # Attack Hyperparameters
    epsilon = 0.05      # Max noise magnitude
    alpha = 0.01        # Step size per iteration
    num_steps = 50      # PGD iterations
    
    print('========================START EXPLOIT & DEFENSE============================')
    for i, sample in enumerate(tqdm(test_loader, desc='Processing Pipeline', total=len(test_loader))):
        low_temp = sample["img16"].numpy()
        low = torch.from_numpy(np.ascontiguousarray(low_temp[:, ::-1, :, :])).to(device)
        
        # --- THE PGD ATTACK LOOP ---
        delta = torch.zeros_like(low, requires_grad=True)
        
        for step in range(num_steps):
            adv_low = low + delta
            adv_sr = net_G_RLS(adv_low)
            adv_sr_resized = F.interpolate(adv_sr, size=(160, 160), mode='bilinear', align_corners=False)
            adv_emb = facenet(adv_sr_resized)
            
            # Minimize distance to target identity
            loss = -F.cosine_similarity(adv_emb, target_emb)
            
            net_G_RLS.zero_grad()
            facenet.zero_grad()
            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()
            
            delta.data = delta.data - alpha * delta.grad.sign()
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)

        # --- EVALUATION AND DEFENSE ROUTINE ---
        with torch.no_grad():
            # 1. The Raw Exploit Images
            final_adv_low = low + delta
            final_adv_sr = net_G_RLS(final_adv_low)
            
            # 2. The Baseline (Control) Image
            baseline_sr = net_G_RLS(low)
            
            # 3. THE PURIFICATION GATEWAY (Defense Execution)
            defended_clean_lr = apply_median_defense(low)
            defended_poisoned_lr = apply_median_defense(final_adv_low)
            
            # 4. Pass the washed images through SCGAN
            defended_baseline_sr = net_G_RLS(defended_clean_lr)
            defended_spoofed_sr = net_G_RLS(defended_poisoned_lr)
            
        # Formatting function for output
        def format_img(tensor):
            img = tensor.cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            return (img * 255).astype(np.uint8)

        img_name = os.path.basename(sample['imgpath'][0]).split('.')[0]
        
        # Save all 6 critical pipeline states
        cv2.imwrite(f"{test_save}/{img_name}_1_original_lr.png", format_img(low))
        cv2.imwrite(f"{test_save}/{img_name}_2_poisoned_lr.png", format_img(final_adv_low))
        cv2.imwrite(f"{test_save}/{img_name}_3_baseline_sr.png", format_img(baseline_sr))
        cv2.imwrite(f"{test_save}/{img_name}_4_spoofed_sr.png", format_img(final_adv_sr))
        cv2.imwrite(f"{test_save}/{img_name}_5_defended_baseline_sr.png", format_img(defended_baseline_sr))
        cv2.imwrite(f"{test_save}/{img_name}_6_defended_spoofed_sr.png", format_img(defended_spoofed_sr))
        
        # Cap at 50 images to match the metric script constraints
        if i >= 49:
            break

if __name__ == '__main__':
    main()