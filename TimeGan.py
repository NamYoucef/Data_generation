import numpy as np
from time_gan_tensorflow.model import TimeGAN
from encoding.encoding_data import load_concatenate_json_files
from encoding.encoding_data import simplify_df
from encoding.encoding_data import encoding_data

# Charger et concaténer les fichiers JSON du dataset
folder_path = './dataset'
df = load_concatenate_json_files(folder_path)

# Nettoyer les données
df = simplify_df(df)

# Encoder les données
real_data = encoding_data(df)

# Diviser les données en train et test
train_size = int(0.8 * len(real_data))

x_train = real_data[:train_size]
x_test = real_data[train_size:]

# Modele TimeGan
model = TimeGAN(
    x=x_train,
    timesteps=20,
    hidden_dim=64,
    num_layers=3,
    lambda_param=0.1,
    eta_param=10,
    learning_rate=0.001,
    batch_size=16
)

# Entrainement
model.fit(
    epochs=100,
    verbose=True
)

# Génération de données
data_hat = model.reconstruct(x=x_test)
data_hat.to_csv('reconstruct_data_TimeGan.csv')

data_sim = model.simulate(samples=len(x_test))
data_sim.to_csv('synthetic_data_TimeGan.csv')