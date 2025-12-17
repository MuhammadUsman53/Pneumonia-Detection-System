import os
import shutil

# This script organizes the standard Kaggle Pneumonia dataset into 3 classes:
# BACTERIA, NORMAL, VIRUS
# Standard Kaggle structure:
# dataset/train/NORMAL
# dataset/train/PNEUMONIA

dataset_dir = 'dataset'
train_dir = os.path.join(dataset_dir, 'train')
pneumonia_dir = os.path.join(train_dir, 'PNEUMONIA')

def organize():
    if not os.path.exists(pneumonia_dir):
        print(f"Directory '{pneumonia_dir}' not found.")
        print("Please ensure you have downloaded the dataset and extracted it into 'dataset/train'.")
        return

    bacteria_dir = os.path.join(train_dir, 'BACTERIA')
    virus_dir = os.path.join(train_dir, 'VIRUS')

    os.makedirs(bacteria_dir, exist_ok=True)
    os.makedirs(virus_dir, exist_ok=True)

    print("Organizing PNEUMONIA images into BACTERIA and VIRUS folders...")
    
    files = os.listdir(pneumonia_dir)
    count_bac = 0
    count_vir = 0
    
    for file in files:
        if 'bacteria' in file.lower():
            shutil.move(os.path.join(pneumonia_dir, file), os.path.join(bacteria_dir, file))
            count_bac += 1
        elif 'virus' in file.lower():
            shutil.move(os.path.join(pneumonia_dir, file), os.path.join(virus_dir, file))
            count_vir += 1
            
    print(f"Moved {count_bac} images to {bacteria_dir}")
    print(f"Moved {count_vir} images to {virus_dir}")
    
    # Check if empty
    if len(os.listdir(pneumonia_dir)) == 0:
        os.rmdir(pneumonia_dir)
        print("Removed empty PNEUMONIA directory.")
        
    print("Done! You can now run 'python train_model.py'")

if __name__ == '__main__':
    organize()
