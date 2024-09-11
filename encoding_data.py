import os
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder

# Load the dataset files
def load_concatenate_json_files(folder_path):
    dfs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                data = json.load(file)
            df = pd.json_normalize(data)
            if 'timestamp' in df.columns and 'actor.mbox' in df.columns and 'verb.display.en' in df.columns and 'object.id' in df.columns:
                df = df[['timestamp', 'actor.mbox', 'verb.display.en', 'object.id']]
                df.columns = ['Timestamp', 'Actor', 'Verb', 'Object']
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# Cleaning the data
def simplify_df(df):
    df['Actor'] = df['Actor'].apply(lambda x: x.split('@')[0]).apply(lambda x: x.replace('mailto:', '').split('@')[0])
    df['Object'] = df['Object'].apply(lambda x: x.replace('http://moodle-example.com/', ''))
    return df

# Enconding data
def encoding_data(df):
    # Apply integer encoding to 'Verb' and 'Object' columns
    label_encoder_actor = LabelEncoder()
    label_encoder_verb = LabelEncoder()
    label_encoder_object = LabelEncoder()
    df['Actor'] = label_encoder_actor.fit_transform(df['Actor'])
    df['Verb'] = label_encoder_verb.fit_transform(df['Verb'])
    df['Object'] = label_encoder_object.fit_transform(df['Object'])
    # Convert the 'Timestamp' column to datetime objects
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Convert datetime timestamps to seconds since the epoch using .timestamp()
    df['Timestamp'] = df['Timestamp'].apply(lambda x: x.timestamp())

    # Find the earliest timestamp in seconds since the epoch
    min_timestamp_seconds = df['Timestamp'].min()

    # Calculate the difference in seconds from the first event
    df['Timestamp'] = df['Timestamp'] - min_timestamp_seconds

    # Sort the DataFrame by the 'Timestamp' column
    df_sorted = df.sort_values(by='Timestamp')

    # Reset the index of the sorted DataFrame to get a new index that represents the chronological order
    df_sorted = df_sorted.reset_index(drop=True)

    # Add the 'Chronological_Order' column at the first position (index 0)
    #df_sorted.insert(0, 'Chronological_Order', df_sorted.index + 1)  # Adding 1 so that the order starts from 1 instead of 0

    df = df_sorted

    # The dataframe is now preprocessed and ready for synthetic data generation
    df_preprocessed_shape = df.shape
    df_preprocessed_head = df.head()

    df_preprocessed_shape, df_preprocessed_head

    return df.values

# formating data
def format_df(df):
    # Convertir la colonne 'Timestamp' en objets datetime pour une manipulation temporelle plus aisée
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Convertir les timestamps datetime en secondes depuis l'époque (epoch) pour une comparaison temporelle uniforme
    df['Timestamp'] = df['Timestamp'].apply(lambda x: x.timestamp())

    # Trouver le timestamp le plus ancien pour normaliser les timestamps par rapport à celui-ci
    min_timestamp_seconds = df['Timestamp'].min()

    # Calculer la différence en secondes par rapport au premier événement pour normaliser les timestamps
    df['Timestamp'] = df['Timestamp'] - min_timestamp_seconds

    # Trier le DataFrame par la colonne 'Timestamp' pour assurer un ordre chronologique
    df_sorted = df.sort_values(by='Timestamp')

    # Réinitialiser l'index du DataFrame trié pour refléter l'ordre chronologique des données
    df_sorted = df_sorted.reset_index(drop=True)

    # Insérer une nouvelle colonne 'Chronological_Order' pour indiquer explicitement l'ordre chronologique des événements
    # df_sorted.insert(0, 'Chronological_Order', df_sorted.index + 1)  # Ajouter 1 pour commencer la commande à partir de 1

    # Affecter le DataFrame prétraité et trié à `df` pour une utilisation future
    df = df_sorted

    # Retourner le DataFrame final
    return df

# Formating timestamp
def format_timestamp(df):
    # Convertir tous les datetimes en un format unique
    df['Timestamp'] = pd.to_datetime(df['Timestamp'],utc=True)
    df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)
    df = df.sort_values(by='Timestamp')
    return df
