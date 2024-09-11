import numpy as np
from sdv.timeseries import PAR
from encoding.encoding_data import load_concatenate_json_files
from encoding.encoding_data import simplify_df
from encoding.encoding_data import format_timestamp


folder_path = './dataset'
df = load_concatenate_json_files(folder_path)
df = simplify_df(df)
df = format_timestamp(df)

# Définir les colonnes d'entité, contexte et index de séquence
entity_columns = []
context_columns = []
sequence_index = 'Timestamp'

# Modele PAR
model = PAR(
    entity_columns=entity_columns,
    context_columns=context_columns,
    sequence_index=sequence_index,
)

# Entrainnement
model.fit(df)

# Génération de données
synthetic_data = model.sample(1)
synthetic_data.to_csv('synthetic_data_PAR.csv')