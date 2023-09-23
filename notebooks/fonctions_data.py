# Version 1.4

import pandas as pd
import os

# -------------------------------------------------------------------------------------------------
# Fonction de création du DF et du DF echantillon
# -------------------------------------------------------------------------------------------------
def import_df(chemin_images, chemin_csv, pourcentage_echantillon = 0.1):
    '''
    Importe le fichier csv et construit 2 df :
        - Le DF basé sur le CSV original
        - Un DF echantillon comportant 10% de données aléatoires du DF original

    Arguments :
        - chemin_images : Chemin vers le dossier images
        - chemin_csv : Chemin vers le fichier .csv contenant les données utilisées
        pourcentage_echantillon : Taille du DF echantillon tiré du DF original
    '''

    # import du df
    df = pd.read_csv(chemin_csv, low_memory=False)
    df['image_url'] = df['image_url'].str.replace('../../images', chemin_images)
    print(f"Nombre d'images chargées pour df: {df.shape[0]}")
    print(f"Nb especes dans df: {df['label'].nunique()}")


    # Contruction de l'echantillon
    L = len(df)
    L_ech = int(pourcentage_echantillon * L)
    df_ech = df.sample(n=L_ech, random_state=10)
    df_ech.reset_index(inplace=True, drop=True)
    print(f"Nombre d'images chargées pour df_ech: {df_ech.shape[0]}")
    print(f"Nb especes dans df_ech: {df_ech['label'].nunique()}")


    return df, df_ech





# -------------------------------------------------------------------------------------------------
# Fonction de controle de la présence des fichiers images
# -------------------------------------------------------------------------------------------------
def controle_presence_fichiers(df, chemin_images):

    '''
    Controle que les fichiers images soient bien présents sur le disque.
        - df : DataFrame contenant les url des fichiers images
        - chemin_images : Variable du DF contenant les url
    '''

    image_directory = chemin_images
    missing_files = []

    # Parcourir chaque ligne du DataFrame
    for index, row in df.iterrows():
        image_path = os.path.join(image_directory, row['image_lien'])

        if not os.path.exists(image_path):
            missing_files.append(image_path)

    # Afficher les fichiers non trouvés
    if missing_files:
        print("\nFichiers non trouvés :")
        for file_path in missing_files:
            print(file_path)

    # Ou préciser que tous les fichiers sont présents
    else:
        print("\nTous les fichiers sont présents.")





# -------------------------------------------------------------------------------------------------
# Fonction d'undersampling des observations
# -------------------------------------------------------------------------------------------------
def undersampling_df(df, col):

    '''
    Undersample le df donné pour équilibrer le nombre d'pobservations par classe.
        - df : df à undersampler
        - col : colonne concernée par le GroupBy pour générer l'undersampling
    '''

    compte = df.groupby(col).count()
    min_samples = compte['image_url'].min()
    min_samples = int(min_samples)

    df_undersample = pd.DataFrame()

    for label, group in df.groupby('label'):
        df_undersample = pd.concat([df_undersample, group.sample(min_samples, replace=True)])
        df_undersample = df_undersample.reset_index(drop=True)

    return df_undersample