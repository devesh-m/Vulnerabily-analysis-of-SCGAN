import os
import cv2
import glob
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_metrics(results_dir=r'D:\ipa_project\SCGAN\attack_results'):
    # Get all the original LR images
    original_lr_files = sorted(glob.glob(os.path.join(results_dir, '*_1_original_lr.png')))
    
    if not original_lr_files:
        print(f"No images found in {results_dir} matching the pattern.")
        return

    # 1. Variables for Stealth (LR vs LR)
    total_ssim_lr = 0
    total_psnr_lr = 0
    
    # 2. Variables for Attack Impact (Baseline SR vs Spoofed SR)
    total_ssim_sr = 0
    total_psnr_sr = 0
    
    # 3. Variables for Defense Success (Baseline SR vs Defended Spoofed SR)
    total_ssim_defense = 0
    total_psnr_defense = 0
    
    # 4. Variables for Utility Penalty (Baseline SR vs Defended Baseline SR)
    total_ssim_utility = 0
    total_psnr_utility = 0
    
    valid_comparisons = 0

    print("==================================================")
    print("  ADVERSARIAL METRICS & DEFENSE AUDIT")
    print("==================================================")

    for orig_lr_path in original_lr_files:
        # Construct the paths for all 6 images in the set
        base_name = orig_lr_path.replace('_1_original_lr.png', '')
        poisoned_lr_path = f"{base_name}_2_poisoned_lr.png"
        baseline_sr_path = f"{base_name}_3_baseline_sr.png"
        spoofed_sr_path = f"{base_name}_4_spoofed_sr.png"
        
        # The new defended images
        defended_baseline_path = f"{base_name}_5_defended_baseline_sr.png"
        defended_spoofed_path = f"{base_name}_6_defended_spoofed_sr.png"
        
        # Check if all files exist for this sample
        if not (os.path.exists(poisoned_lr_path) and os.path.exists(baseline_sr_path) and 
                os.path.exists(spoofed_sr_path) and os.path.exists(defended_baseline_path) and 
                os.path.exists(defended_spoofed_path)):
            continue

        # Load images
        img_orig_lr = cv2.imread(orig_lr_path)
        img_poison_lr = cv2.imread(poisoned_lr_path)
        img_base_sr = cv2.imread(baseline_sr_path)
        img_spoof_sr = cv2.imread(spoofed_sr_path)
        img_defended_base_sr = cv2.imread(defended_baseline_path)
        img_defended_spoof_sr = cv2.imread(defended_spoofed_path)

        # 1. Calculate Stealth (Input layer)
        ssim_lr = ssim(img_orig_lr, img_poison_lr, channel_axis=-1, win_size=3, data_range=255)
        psnr_lr = psnr(img_orig_lr, img_poison_lr, data_range=255)
        
        # 2. Calculate Damage (Unprotected Output)
        ssim_sr = ssim(img_base_sr, img_spoof_sr, channel_axis=-1, win_size=7, data_range=255)
        psnr_sr = psnr(img_base_sr, img_spoof_sr, data_range=255)
        
        # 3. Calculate Defense Success (Did it fix the poisoned image?)
        ssim_defense = ssim(img_base_sr, img_defended_spoof_sr, channel_axis=-1, win_size=7, data_range=255)
        psnr_defense = psnr(img_base_sr, img_defended_spoof_sr, data_range=255)
        
        # 4. Calculate Utility Penalty (Did it ruin the clean image?)
        ssim_utility = ssim(img_base_sr, img_defended_base_sr, channel_axis=-1, win_size=7, data_range=255)
        psnr_utility = psnr(img_base_sr, img_defended_base_sr, data_range=255)

        total_ssim_lr += ssim_lr
        total_psnr_lr += psnr_lr
        total_ssim_sr += ssim_sr
        total_psnr_sr += psnr_sr
        total_ssim_defense += ssim_defense
        total_psnr_defense += psnr_defense
        total_ssim_utility += ssim_utility
        total_psnr_utility += psnr_utility
        valid_comparisons += 1
        
        file_id = os.path.basename(base_name)
        print(f"Sample: {file_id} | Attack SSIM: {ssim_sr:.4f} | Defense SSIM: {ssim_defense:.4f} | Utility SSIM: {ssim_utility:.4f}")

    if valid_comparisons > 0:
        avg_ssim_lr = total_ssim_lr / valid_comparisons
        avg_psnr_lr = total_psnr_lr / valid_comparisons
        avg_ssim_sr = total_ssim_sr / valid_comparisons
        avg_psnr_sr = total_psnr_sr / valid_comparisons
        avg_ssim_defense = total_ssim_defense / valid_comparisons
        avg_psnr_defense = total_psnr_defense / valid_comparisons
        avg_ssim_utility = total_ssim_utility / valid_comparisons
        avg_psnr_utility = total_psnr_utility / valid_comparisons
        
        print("\n================ FINAL PAPER RESULTS ===================")
        print(f"Images Evaluated: {valid_comparisons}")
        print(f"\n--- 1. INPUT STEALTH (Original LR vs Poisoned LR) ---")
        print(f"Average SSIM: {avg_ssim_lr:.4f} (Closer to 1.0 means perfectly invisible)")
        print(f"Average PSNR: {avg_psnr_lr:.2f} dB")
        
        print(f"\n--- 2. OUTPUT DAMAGE (Baseline SR vs Spoofed SR) ---")
        print(f"Average SSIM: {avg_ssim_sr:.4f} (Lower score means successful attack)")
        print(f"Average PSNR: {avg_psnr_sr:.2f} dB")
        
        print(f"\n--- 3. DEFENSE SUCCESS (Baseline SR vs Defended Spoof SR) ---")
        print(f"Average SSIM: {avg_ssim_defense:.4f} (Should jump back up near 0.90+)")
        print(f"Average PSNR: {avg_psnr_defense:.2f} dB")
        
        print(f"\n--- 4. UTILITY PENALTY (Baseline SR vs Defended Baseline SR) ---")
        print(f"Average SSIM: {avg_ssim_utility:.4f} (Should be very high, proving normal faces aren't ruined)")
        print(f"Average PSNR: {avg_psnr_utility:.2f} dB")
        print("========================================================")

if __name__ == '__main__':
    calculate_metrics()