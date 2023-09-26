# Version 1.4

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input


# -------------------------------------------------------------------------------------------------
# Fonction de creation d'un dataset Tensorflow
# -------------------------------------------------------------------------------------------------
def create_tf_dataset(image_path, img_dim, labels, batch_size, augment = False):
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
    dataset = dataset.map(lambda image_path, label: augment_img(image_path, label, img_dim, augment),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Mélange aléatoire du dataset
    dataset = dataset.shuffle(buffer_size=len(image_path))

    # Découpage en batch
    dataset = dataset.batch(batch_size)

    # Optimisation : Charge les données en arrière-plan et maintien la charge CPU/GPU
    #dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


    return dataset





# -------------------------------------------------------------------------------------------------
# Fonction d'augmentation des images
# -------------------------------------------------------------------------------------------------
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