import warnings
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image

warnings.filterwarnings("ignore", category=UserWarning) # Pentru încetarea afișării unor avertismente minore

model_path = "../models/BEEEEEST_BEST_model_functional_categorical_padding_2506_1436.keras" # Calea către modelul salvat
img_path = "../data/processed/test/lung_scc/lungscc1516.jpeg" # Imaginea asupra căreia se aplică Grad-CAM
IMG_SIZE = (64, 64) # Dimensiunea standard folosită de CNN

filename = os.path.basename(img_path) # Extragerea numelui fișierului din cale

# Încărcarea modelului CNN
model = tf.keras.models.load_model(model_path)

# Încărcare imagine
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array / 255.0, axis=0)

# Predicția clasei
preds = model.predict(img_array)
class_labels = ['lung_aca', 'lung_n', 'lung_scc', 'colon_aca', 'colon_n']
pred_class = np.argmax(preds[0]) # Extragere clasă cu probabilitatea maximă

# Afișare clasă prezisă și scor de încredere
print(f"Predicted class index: {pred_class} ({class_labels[pred_class]}) - (Confidence: {preds[0][pred_class]:.4f})")

# Identificare ultimului strat convoluțional din CNN
last_conv_layer = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer.name
        break

# Verificare dacă ultimul start convoluțional a fost găsit
if last_conv_layer is None:
    raise ValueError("Nu s-a gasit niciun strat Conv2D in modelul incarcat!")

# Creare model auxiliar pt. extragerea activărilor
grad_model = models.Model(
    [model.inputs], [model.get_layer(last_conv_layer).output, model.output]
)

# Calcul derivare gradient față de clasa prezisă
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[:, pred_class]

# Calcul gradient mediu per canal
grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
conv_outputs = conv_outputs[0]

# Generare heatmap
heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

# Citire imagine originală pt. suprapunere heatmap
img_orig = cv2.imread(img_path)
orig_size = (img_orig.shape[1], img_orig.shape[0])

# Normalizare heatmap
heatmap = np.maximum(heatmap, 0)
heatmap_max = np.max(heatmap)

if heatmap_max == 0 or np.isnan(heatmap_max) or np.isinf(heatmap_max):
    heatmap = np.zeros_like(heatmap)
else:
    heatmap = heatmap / heatmap_max

# Redimensionare heatmap
heatmap = cv2.resize(heatmap, orig_size)
heatmap = np.clip(heatmap, 0, 1)
heatmap = np.uint8(255 * heatmap)

# Aplicare colormap și suprapunere peste imaginea originală
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
superimposed = cv2.addWeighted(img_orig, 0.5, heatmap_color, 0.5, 0)

# Salvare imagine Grad-CAM
cv2.imwrite("../results/gradcam_lungscc1516.png", superimposed)

# Afișare imagine originală și imagine Grad-CAM
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title(f"Imaginea originală\n{filename}")
plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title(f"Afișarea Grad-CAM\n{filename}")
plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.tight_layout()
plt.show()