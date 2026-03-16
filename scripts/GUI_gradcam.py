import os
from tkinter import filedialog
import customtkinter as ctk
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "../models/BEEEEEST_BEST_model_functional_categorical_padding_2506_1436.keras" # Calea către modelul salvat
CLASS_LABELS = ['Colon - adenocarcinoma', 'Colon - benign', 'Lung - adenocarcinoma', 'Lung - benign', 'Lung - squamous cell carcinoma'] # Clasele pe care le poate prezice modelul
IMG_SIZE = (64, 64) # Dimensiunea la care sunt redimensionate imaginile
model = load_model(MODEL_PATH) # Încărcare model antrenat

# CLASA PRINCIPALĂ A APLICAȚIEI - definește aplicația CustomTkinter
class GradCAMClassifierApp(ctk.CTk):

    # Constructor - setup GUI
    def __init__(self):
        super().__init__()

        #Setări generale fereastră
        self.title("Clasificare imagini și Grad-CAM")
        self.state("zoomed")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        # Container butoane control
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.pack(pady=10)

        #Buton încărcare imagini
        self.load_button = ctk.CTkButton(self.button_frame, text=">>Încarcă imagini", command=self.load_images, font=ctk.CTkFont(size=16, weight="bold"))
        self.load_button.grid(row=0, column=0, padx=20)

        # Buton clasificare
        self.classify_button = ctk.CTkButton(self.button_frame, text=">>Clasifică", command=self.classify_images, font=ctk.CTkFont(size=16, weight="bold"))
        self.classify_button.grid(row=0, column=1, padx=20)

        # Buton afișare GradCAM
        self.gradcam_button = ctk.CTkButton(self.button_frame, text=">>Afișează Grad-CAM", command=self.show_gradcams, font=ctk.CTkFont(size=16, weight="bold"))
        self.gradcam_button.grid(row=0, column=2, padx=20)

        # Zonă pentru imagini cu scroll
        self.display_frame = ctk.CTkScrollableFrame(self, width=1200, height=700)
        self.display_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Liste pt. imagini și referințe
        self.image_paths = []
        self.image_widgets = []

    # FUNCȚIE ÎNCĂRCARE IMAGINI
    def load_images(self):

        # Deschidere fereastră de selecție
        self.image_paths = filedialog.askopenfilenames(filetypes=[("Imagini", "*.jpg *.jpeg *.png")])
        self.clear_display()
        self.image_widgets.clear()

        # Creare CTkImage pt. afișare imagini în interfață
        for i, path in enumerate(self.image_paths):
            img = Image.open(path)
            img_ctk = ctk.CTkImage(light_image=img, size=(250, 250))

            # Creare row_frame cu imaginea originală
            row_frame = ctk.CTkFrame(self.display_frame)
            row_frame.pack(padx=10, pady=10, fill="x")

            # Creare label cu numele fișierului
            image_label = ctk.CTkLabel(row_frame, text="", image=img_ctk)
            image_label.grid(row=0, column=0, padx=10)
            filename = os.path.basename(path)
            label_widget = ctk.CTkLabel(row_frame, text=f"{filename}", font=ctk.CTkFont(size=14))
            label_widget.grid(row=0, column=1, padx=10)

            # Păstrare referință pt. afișare clasificare și GradCAM
            self.image_widgets.append((img_ctk, label_widget, None))

    # FUNCȚIE CLASIFICARE IMAGINI
    def classify_images(self):
        for i, path in enumerate(self.image_paths):

            # Încărcare, redimensionare și normalizare imagine
            img = image.load_img(path, target_size=IMG_SIZE)
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predicția modelului încărcat
            preds = model(img_array, training=False).numpy()
            class_index = np.argmax(preds[0])
            confidence = preds[0][class_index]
            label = CLASS_LABELS[class_index]

            # Afișare eticheta prezisă și incredere
            text = f"{os.path.basename(path)}\nClasă: {label}\nÎncredere: {confidence:.2%}"
            color = "green" if confidence > 0.85 else "orange" if confidence > 0.6 else "red" # etichetă colorată în funcție de grad de încredere

            self.image_widgets[i][1].configure(text=text, text_color=color)

    # FUNCȚIE AFIȘARE GRAD-CAM
    def show_gradcams(self):

        # Generare Grad-CAM pt. fiecare imagine
        for i, path in enumerate(self.image_paths):
            gradcam_img = self.generate_gradcam(path)
            gradcam_img = cv2.cvtColor(gradcam_img, cv2.COLOR_BGR2RGB)
            gradcam_pil = Image.fromarray(gradcam_img)
            gradcam_ctk = ctk.CTkImage(light_image=gradcam_pil, size=(250, 250))

            # Creare nou label in acelasi rand (afișare în dreptul imaginii originale)
            gradcam_label = ctk.CTkLabel(self.display_frame.winfo_children()[i], image=gradcam_ctk, text="")
            gradcam_label.grid(row=0, column=2, padx=10)

            self.image_widgets[i] = (*self.image_widgets[i][:2], gradcam_ctk)

    # FUNCȚIE GENERARE GRAD-CAM
    def generate_gradcam(self, img_path):

        #Preprocesare imagine
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model(img_array, training=False).numpy()
        class_index = np.argmax(preds[0])

        # Identificare ultimul start convoluțional
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
                break

        # Obținere caracteristici de ieșire de pe ultimul start convoluțional
        # Creare model intermediar
        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv).output, model.output])

        # Calcul gradient față de clasa prezisă
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_index]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        # Calcul hartă de activare (heatmap)
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap_max = np.max(heatmap)
        if heatmap_max > 0:
            heatmap /= heatmap_max

        img_orig = cv2.imread(img_path)
        orig_size = (img_orig.shape[1], img_orig.shape[0])

        heatmap_max = np.max(heatmap)
        print(f"[DEBUG] Heatmap max value for {os.path.basename(img_path)}: {heatmap_max}")

        heatmap = cv2.resize(heatmap, orig_size)
        heatmap = np.uint8(255 * np.clip(heatmap, 0, 1))
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)

        # Suprapunere heatmap peste imaginea originală
        superimposed = cv2.addWeighted(img_orig, 0.5, heatmap_color, 0.5, 0)

        # Returnare afișare GradCAM
        return superimposed

    # FUNCȚIE CURĂȚARE ZONĂ DE AFIȘARE
    def clear_display(self):
        for widget in self.display_frame.winfo_children():
            widget.destroy()

# RULARE APLICAȚIE - creează și lansează fereastra principală
if __name__ == "__main__":
    app = GradCAMClassifierApp()
    app.mainloop()
