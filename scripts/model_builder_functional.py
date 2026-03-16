from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def build_model(input_shape=(64, 64, 3), num_classes=5): # Definește funcția care va crea modelul CNN (dimensiune imagini intrare, număr de claase)
    inputs = Input(shape=input_shape, name="input_layer") # Definește stratul de intrare

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs) # Primul strat convoluțional (32 măști 3x3, activare ReLU, padding)
    x = MaxPooling2D(pool_size=(2, 2))(x) # Primul strat pooling (reducere dimensiuni 64x64 -> 32x32, extragere caracteristici)
    x = MaxPooling2D(pool_size=(2, 2))(x) # Al doilea strat pooling (reducere dimensiuni 32x32 -> 16x16)
    x = Conv2D(128, (3, 3), activation='relu')(x) # Al doilea strat convoluțional (128 măști 3x3, activare ReLU, fără padding -> scade dimensiunea 16x16 -> 14x14)
    x = MaxPooling2D(pool_size=(2, 2))(x) # Al treilea strat pooling (reducere dimensiuni 14x14 -> 7x7)
    x = BatchNormalization()(x) # Normalizare caracteristici (stabilizare antrenare, accelerare convergență, reducere risc overfitting)
    x = MaxPooling2D(pool_size=(2, 2))(x) # Al patrulea strat pooling (reducere dimensiuni 7x7 -> 3x3)
    x = Dropout(0.5)(x) # Dezactivare aleatorie 50% din neuroni/epocă (prevenire overfitting)
    x = Flatten()(x) # Transformare matrice 3D în vector 1D (necesar pentru dense)
    x = Dense(128, activation='relu')(x) # Strat complet conectat cu 128 neuroni, activare ReLU (învățare caracteristici complexe din convoluții)

    outputs = Dense(num_classes, activation='softmax')(x) # Stratul final/de ieșire (nr. neuroni = nr. clase, activare Softmax -> returnare probabilități pt. fiecare clasă)

    model = Model(inputs=inputs, outputs=outputs) # Construire model final
    return model # Returnare model
