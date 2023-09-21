# Version 1.3




import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input




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







def create_tf_dataset(image_path, img_dim, labels, batch_size, augment = True):
    '''
    Créé un dataset Tensorflow selon les paramètres précisés.
        - image_path : chemin relatif de la variable contenant les images
        - labels : variable contenant les labels
        - batch_size : taille des batchs
        - augment : True pour agumentation d'image
        - img_dim : Dimensions des images
    '''

    #image_path = image_path.tolist()  # Convertir les chemins d'images en liste
    #labels = labels.tolist()          # Convertir les labels en liste


    # Construction du Dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_path, labels))

    # Application de l'augmentation d'images
    #dataset = dataset.map(augment_img(image_path, labels, img_dim, augment), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda image_path, label: augment_img(image_path, label, img_dim, augment),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Mélange aléatoire du dataset
    dataset = dataset.shuffle(buffer_size=len(image_path))

    # Découpage en batch
    dataset = dataset.batch(batch_size)

    # Optimisation : Charge les données en arrière-plan et maintien la charge CPU/GPU
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


    return dataset




def augment_img(image_path, labels, img_dim, augment):

    '''
    Modifie les images aléatoirement dans le dataset qui sera soumis au modèle.
      - image_path : URL des images (contenue dans la variable 'image_url' dans le DF chargé),
      - label : Variable contenant les classes
      - img_dim : Dimensions des images
      - augment : True pour augmentation d'images
    '''

    # Lecture image, decodage
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)

    # Redimensionnement selon dimensions définies en début de notebook (img_dim)
    img = tf.image.resize(img, img_dim)


    if augment == True:

        # Pre-processing pour transfert learning, modèle efficienNet
        img = preprocess_input(img)

        # Augmentations aléatoires des images :

        # Inversion Gauche/Droite
        img = tf.image.random_flip_left_right(img)

        # Inversion Haut/Bas
        img = tf.image.random_flip_up_down(img)

        # Modification luminosité
        img = tf.image.random_brightness(img, max_delta=0.2)

        # Modification contraste
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)



    # Conversion du type en float32
    img = tf.image.convert_image_dtype(img, tf.float32)

    # Normalisation
    img = (img - tf.math.reduce_min(img)) / (tf.math.reduce_max(img) - tf.math.reduce_min(img))

    return img, labels