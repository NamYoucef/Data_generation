from encoding.encoding_data import load_concatenate_json_files
from encoding.encoding_data import simplify_df
from encoding.encoding_data import format_df
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata



folder_path = './dataset'
# Charger les données dans un dataframe
df = load_concatenate_json_files(folder_path)

# Nettoyer les données
df = simplify_df(df)

# Formater les données
df = format_df(df)

# Définir les metadata et les contraintes
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)
metadata.visualize()

# Modele CTGAN
ctgan = CTGANSynthesizer(metadata, epochs=200)
# Entrainnement
ctgan.fit(df)

# Génération de données
synthetic_data = ctgan.sample(len(df))
synthetic_data.to_csv('synthetic_data_CTGAN.csv')
