import cv2
import numpy as np
import os
import glob
import shutil
import yaml

# Configuration
SOURCE_DIR = r"e:\Blue_Ocean\SSDD.v1i.yolov8"
OUTPUT_DIR = r"e:\Blue_Ocean\SSDD_OBB"
SPLITS = ['train', 'valid', 'test']

def convert_polygon_to_obb(poly_coords):
    """
    Convert normalized polygon coordinates to normalized OBB coordinates (x1, y1, x2, y2, x3, y3, x4, y4).
    """
    # Convert to numpy array of points
    points = np.array(poly_coords, dtype=np.float32).reshape(-1, 2)
    
    # We need to denormalize temporarily to compute minAreaRect correctly (aspect ratio matters)
    # However, if we assume square pixels, we can compute on normalized 0-1 space, 
    # BUT rotation might be skewed if image is non-square.
    # It is safer to use pixel coordinates if possible, but we might not strictly need to if we handle it carefully.
    # Let's rely on the image dimensions loop below.
    return points

def process_dataset():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory {SOURCE_DIR} not found.")
        return

    print(f"Converting dataset from {SOURCE_DIR} to {OUTPUT_DIR}...")
    
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    for split in SPLITS:
        src_images = os.path.join(SOURCE_DIR, split, 'images')
        src_labels = os.path.join(SOURCE_DIR, split, 'labels')
        
        dst_images = os.path.join(OUTPUT_DIR, split, 'images')
        dst_labels = os.path.join(OUTPUT_DIR, split, 'labels')
        
        os.makedirs(dst_images, exist_ok=True)
        os.makedirs(dst_labels, exist_ok=True)
        
        # Get all label files
        label_files = glob.glob(os.path.join(src_labels, "*.txt"))
        
        for label_path in label_files:
            file_name = os.path.basename(label_path)
            image_name = file_name.replace('.txt', '.jpg') # Assuming jpg
            image_path = os.path.join(src_images, image_name)
            
            if not os.path.exists(image_path):
                # Try png
                image_name = file_name.replace('.txt', '.png')
                image_path = os.path.join(src_images, image_name)
                if not os.path.exists(image_path):
                    print(f"Warning: Image for {file_name} not found. Skipping.")
                    continue
            
            # Copy image
            shutil.copy2(image_path, os.path.join(dst_images, image_name))
            
            # Read image for dimensions
            img = cv2.imread(image_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            
            # Process label
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = list(map(float, line.strip().split()))
                class_id = int(parts[0])
                coords = parts[1:]
                
                # Denormalize
                points = []
                for i in range(0, len(coords), 2):
                    points.append([coords[i] * w, coords[i+1] * h])
                
                points_np = np.array(points, dtype=np.float32)
                
                # minAreaRect
                rect = cv2.minAreaRect(points_np)
                box = cv2.boxPoints(rect)
                # Keep as float for precision instead of np.int0
                
                # Normalize back
                norm_box = []
                for p in box:
                    norm_box.append(p[0] / w)
                    norm_box.append(p[1] / h)
                
                # Clamp to 0-1
                norm_box = [min(max(x, 0.0), 1.0) for x in norm_box]
                
                # Format: class x1 y1 x2 y2 x3 y3 x4 y4
                line_str = f"{class_id} " + " ".join([f"{x:.6f}" for x in norm_box])
                new_lines.append(line_str)
            
            with open(os.path.join(dst_labels, file_name), 'w') as f:
                f.write("\n".join(new_lines))
                
    print("Conversion complete.")
    
    # Create config file
    config_content = f"""path: {OUTPUT_DIR}
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['ship']
"""
    with open(os.path.join(OUTPUT_DIR, "data_obb.yaml"), "w") as f:
        f.write(config_content)
    print("Created data_obb.yaml")

if __name__ == "__main__":
    process_dataset()
