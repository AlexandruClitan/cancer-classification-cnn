import os
import pickle
import time
import tensorflow as tf

from datetime import datetime

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from scripts.model_builder_functional import build_model

timestamp = datetime.now().strftime("%d%m_%H%M") # Creare string data și oră pt. numele modelelor salvate

# Căi către subdirectoarele setului de date preprocesat
base_dir = "../data/processed"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# Parametri pentru imaginile de intrare
IMG_HEIGHT = 64 # Înălțime imagine
IMG_WIDTH = 64 # Lățime imagine
BATCH_SIZE = 32 # Nr. imagini procesate simultan la antrenare

# Augmentare date pt. antrenare -> creștere variație date
train_datagen = ImageDataGenerator(
    rescale=1. / 255, # Normalizare pixeli (între 0 și 1)
    rotation_range=5, # Rotire aleatorie a imaginilor (-5/+5 grade)
    zoom_range=0.05, # Mărire/micșorare imagini (1 - zoom_factor sau 1 + zoom_factor) -> [0.95, 1.05]
    horizontal_flip=True # Răsturnare orizontală (pe axa Y) a imaginilor în mod aleatoriu
)

# Normalizare pentru imaginile de validare și test
val_test_datagen = ImageDataGenerator(rescale=1./255) # Scalare

# Creare generatoare pentru cele 3 subseturi
# Generator set de antrenare (imagini intrare)
train_generator = train_datagen.flow_from_directory(
    train_dir, # Directorul cu imagini pt. antrenare
    target_size=(IMG_HEIGHT, IMG_WIDTH), # Redimensionare imagini înainte de a fi furnizate modelului
    batch_size=BATCH_SIZE, # Generatorul furnizează modelului 32 de imagini odată
    class_mode='categorical' # generare etichete de clasă în format one-hot encoded
)

# Generator set de validare
val_generator = val_test_datagen.flow_from_directory(
    val_dir, # Directorul cu imagini pt. validare
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Generator set de test
test_generator = val_test_datagen.flow_from_directory(
    test_dir, # Directorul cu imagini pt. test
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False # Asigurarea corespondenței între predicții și etichete
)

# Afișare clase model (mapping-ul etichetelor)
print("Clasele modelului (etichetate):", train_generator.class_indices)

# Creare model CNN
model = build_model(input_shape=(64, 64, 3), num_classes=5)

# Compilare model CNN (modul în care învață)
model.compile(optimizer=Adam(learning_rate=0.0005), # Algoritmul de optimizare
              loss='categorical_crossentropy', # Funcția de pierdere
              metrics=['accuracy']) # Metricile de performnță urmărite -> acuratete

# Salvarea modelului în versiunea cea mai performantă
best_model_filename = f"../models/best_model_functional_categorical_padding_{timestamp}.keras"

# Metode de callback
# Salvare DOAR cel mai bun model (val_loss minim)
best_model_checkpoint = ModelCheckpoint(
    filepath=best_model_filename,
    monitor='val_loss',
    save_best_only=True,
    verbose=1 #afisare bara de progres detaliata in timpul antrenarii
)

# Oprire proces antrenare dacă nu există îmbunătățiri
early_stopping = EarlyStopping(
    monitor='val_loss', # Monitorizare pierderi pe validare
    patience=7, # c
    restore_best_weights=True, # Revenire la parametrii din epoca cu cea mai bună performanță
    verbose=1
)

# Reducere rată de învățare dacă performanța stagnează
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', # Monitorizare pierderi pe validare
    factor=0.5, # Rata se înjumătățește (-50%)
    patience=2, # Daca timp de 2 epoci nu se îmbunătățește val_loss
    min_lr=1e-6, # Valoarea minimă până la care poate scade rata
    verbose=1
)

# Lista finală de callbacks
callbacks = [early_stopping, best_model_checkpoint, reduce_lr]

# Acordarea unor ponderi mai mari claselelor între care există confuzie
class_weight = {
    0: 1.0, #colon_aca
    1: 1.0, #colon_n
    2: 1.5, #lung_aca - confuzie
    3: 1.0, #lung_n
    4: 1.7  #lung_scc - confuzie
}

# Start cronometru pentru antrenare
start_time = time.time()

# Antrenare model -> returnează pierderea la antrenare pe epoci, pierderea la validare și acuratețea
history = model.fit( # Funcție antrenare
    train_generator, # Date intrare
    validation_data=val_generator, # Date de validare (pe fiecare epocă)
    epochs=60, # Număr epoci
    callbacks=callbacks, # Callback-uri
    class_weight = class_weight # Ponderi pe clase
)

# Stop cronometru pentru antrenare
end_time = time.time()

# Afișarea timpului total de antrenare (minute)
print(f"Timp total antrenare: {(end_time - start_time)/60:.2f} minute")

# Salvarea istoricului de antrenare (loss, accuracy, val_loss, val_accuracy etc.)
with open("../models/training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

# Evaluare pe setul de antrenare  - acuratețe și pierdere
loss, accuracy = model.evaluate(train_generator)
print(f"Train accuracy: {accuracy:.4f}")

# Evaluare pe setul de test - acuratețe și pierdere
loss, accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {accuracy:.4f}")

# Salvarea rezultatelor testului, timpul de antrenare, timestamp etc.
with open("../results/best_model_log.txt", "a") as log_file:
    log_file.write(f"Timestamp: {timestamp} | Test accuracy: {accuracy:.4f} | Test loss: {loss:.4f} | Train time: {(end_time - start_time)/60:.2f} min\n")

# Salvare model complet
model.save(f"../models/model_functional_categorical_padding_{timestamp}.keras")