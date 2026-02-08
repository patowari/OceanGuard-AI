from ultralytics import YOLO
import torch

def train_on_cpu():
    """
    Sets up and runs the YOLOv8-OBB training on a local CPU.
    """
    # Verify CPU is being used
    print(f"PyTorch device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    # Load a pretrained OBB model
    # This will download 'yolov8s-obb.pt' if it's not already cached
    model = YOLO('yolov8s-obb.pt')

    print("Starting model training on CPU. This will take a significant amount of time.")

    # Train the model
    try:
        results = model.train(
            data=r'e:\Blue_Ocean\SSDD_OBB\data_obb.yaml',  # Absolute path to the dataset config
            epochs=10,           # Reduced for feasible CPU training time
            imgsz=640,           # Image size
            batch=8,             # Smaller batch size for CPU to manage memory
            name='aegis_ocean_model_cpu', # A new name for the CPU-trained model
            device='cpu',        # Explicitly set to CPU
            exist_ok=True        # Allows re-running without errors
        )
        print("Training finished successfully.")
        print("Model and results are saved in the 'runs/detect/aegis_ocean_model_cpu' directory.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        print("Please ensure 'ultralytics' is installed (`pip install ultralytics`).")

if __name__ == '__main__':
    train_on_cpu()
