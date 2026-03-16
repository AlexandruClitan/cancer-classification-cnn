import os
import shutil
import random

# Definire căi pentru imaginile brute
source_dir = "../data/raw" # Directorul cu imagini grupate pe clase (nesortate)
destination_dir = "../data/processed" # Directorul în care vor fi copiate imaginile, organizate pe seturi

# Numele claselor (subfoldere din raw/)
categories = ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]

# Proporții pt. împărțire
split_ratios = {
    "train": 0.7, # 70% antrenare
    "val": 0.15, # 15% validare
    "test": 0.15 # 15% testare
}

# Creare automată directoare pt. fiecare subset și clasă
for split in split_ratios: # Parcurge 'train', 'val', 'test'
    for category in categories: # Parcurge fiecare clasă
        os.makedirs(os.path.join(destination_dir, split, category), exist_ok=True) # Creează directoarele necesare în 'processed'

# Împărțire și copiere fișiere
for category in categories: # Procesare pt. fiecare clasă
    cat_path = os.path.join(source_dir, category) # Creează calea completă spre folderul clasei
    images = os.listdir(cat_path) # Listare toate fișierele din acel folder
    random.shuffle(images) # Amestecare aleatorie

    total = len(images) # Nr. total de imagini în acea clasă
    train_end = int(split_ratios["train"] * total) # Indexul până la care sunt imaginile de antrenare
    val_end = train_end + int(split_ratios["val"] * total) # Indexul până la care sunt imaginile de validare
    # Restul sunt imaginile de test

    # Împărțirea listelor de imagini în subseturi
    splits = {
        "train": images[:train_end], # Primele 70%
        "val": images[train_end:val_end], # Următoarele 15%
        "test": images[val_end:] # Restul de 15%
    }

    # Copierea fișierelor în folderele corespunzătoare
    for split, split_images in splits.items(): # Parcurge fiecare subset
        for img in split_images: # Parcurge imaginile din acel subset
            src = os.path.join(cat_path, img) # Calea completă a imaginii originale
            dst = os.path.join(destination_dir, split, category, img) # Destinația imaginii
            shutil.copyfile(src, dst) # Copiază imaginea de la sursă la destinație

# Mesaj de confirmare
print(" Dataset-ul a fost împărțit cu succes în train, val și test.")
