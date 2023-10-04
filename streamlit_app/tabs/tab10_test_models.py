import streamlit as st
import itertools
import os

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import zipfile
import glob

import tensorflow as tf
import tensorflow_hub as hub

title = "Test a model"
sidebar_name = "Test a model"

# All path and file names are hard coded for now
def unzip_images():
    zip_file = './assets/imgFor_test.zip'
    with zipfile.ZipFile(zip_file,"r") as z:
        z.extractall("./assets")
    return "./assets/imgFor_test"

def build_dataset(data_dir, IMAGE_SIZE):
  return tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      label_mode="categorical",
      image_size=IMAGE_SIZE,
      batch_size=1)

def run():

    st.title(title)

    st.write("TF version:", tf.__version__)
    st.write("TF Hub version:", hub.__version__)
    st.write("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

    st.markdown(
        """
        Choose a model to train and evaluate.
        """
    )


    # model_name dict
    model_handle_map = {
    "efficientnetv2-s": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/feature_vector/2",
    "efficientnetv2-m": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_m/feature_vector/2",
    "efficientnetv2-l": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_l/feature_vector/2",
    "efficientnetv2-s-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_s/feature_vector/2",
    "efficientnetv2-m-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_m/feature_vector/2",
    "efficientnetv2-l-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_l/feature_vector/2",
    "efficientnetv2-xl-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/feature_vector/2",
    "efficientnetv2-b0-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/feature_vector/2",
    "efficientnetv2-b1-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b1/feature_vector/2",
    "efficientnetv2-b2-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b2/feature_vector/2",
    "efficientnetv2-b3-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b3/feature_vector/2",
    "efficientnetv2-s-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_s/feature_vector/2",
    "efficientnetv2-m-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_m/feature_vector/2",
    "efficientnetv2-l-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_l/feature_vector/2",
    "efficientnetv2-xl-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/feature_vector/2",
    "efficientnetv2-b0-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b0/feature_vector/2",
    "efficientnetv2-b1-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b1/feature_vector/2",
    "efficientnetv2-b2-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b2/feature_vector/2",
    "efficientnetv2-b3-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b3/feature_vector/2",
    "efficientnetv2-b0": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2",
    "efficientnetv2-b1": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b1/feature_vector/2",
    "efficientnetv2-b2": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b2/feature_vector/2",
    "efficientnetv2-b3": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b3/feature_vector/2",
    "efficientnet_b0": "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1",
    "efficientnet_b1": "https://tfhub.dev/tensorflow/efficientnet/b1/feature-vector/1",
    "efficientnet_b2": "https://tfhub.dev/tensorflow/efficientnet/b2/feature-vector/1",
    "efficientnet_b3": "https://tfhub.dev/tensorflow/efficientnet/b3/feature-vector/1",
    "efficientnet_b4": "https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1",
    "efficientnet_b5": "https://tfhub.dev/tensorflow/efficientnet/b5/feature-vector/1",
    "efficientnet_b6": "https://tfhub.dev/tensorflow/efficientnet/b6/feature-vector/1",
    "efficientnet_b7": "https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1",
    "bit_s-r50x1": "https://tfhub.dev/google/bit/s-r50x1/1",
    "inception_v3": "https://tfhub.dev/google/imagenet/inception_v3/feature-vector/4",
    "inception_resnet_v2": "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature-vector/4",
    "resnet_v1_50": "https://tfhub.dev/google/imagenet/resnet_v1_50/feature-vector/4",
    "resnet_v1_101": "https://tfhub.dev/google/imagenet/resnet_v1_101/feature-vector/4",
    "resnet_v1_152": "https://tfhub.dev/google/imagenet/resnet_v1_152/feature-vector/4",
    "resnet_v2_50": "https://tfhub.dev/google/imagenet/resnet_v2_50/feature-vector/4",
    "resnet_v2_101": "https://tfhub.dev/google/imagenet/resnet_v2_101/feature-vector/4",
    "resnet_v2_152": "https://tfhub.dev/google/imagenet/resnet_v2_152/feature-vector/4",
    "nasnet_large": "https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/4",
    "nasnet_mobile": "https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/4",
    "pnasnet_large": "https://tfhub.dev/google/imagenet/pnasnet_large/feature_vector/4",
    "mobilenet_v2_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4",
    "mobilenet_v2_130_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/4",
    "mobilenet_v2_140_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4",
    "mobilenet_v3_small_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5",
    "mobilenet_v3_small_075_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_075_224/feature_vector/5",
    "mobilenet_v3_large_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5",
    "mobilenet_v3_large_075_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/feature_vector/5",
    }

    model_image_size_map = {
    "efficientnetv2-s": 384,
    "efficientnetv2-m": 480,
    "efficientnetv2-l": 480,
    "efficientnetv2-b0": 224,
    "efficientnetv2-b1": 240,
    "efficientnetv2-b2": 260,
    "efficientnetv2-b3": 300,
    "efficientnetv2-s-21k": 384,
    "efficientnetv2-m-21k": 480,
    "efficientnetv2-l-21k": 480,
    "efficientnetv2-xl-21k": 512,
    "efficientnetv2-b0-21k": 224,
    "efficientnetv2-b1-21k": 240,
    "efficientnetv2-b2-21k": 260,
    "efficientnetv2-b3-21k": 300,
    "efficientnetv2-s-21k-ft1k": 384,
    "efficientnetv2-m-21k-ft1k": 480,
    "efficientnetv2-l-21k-ft1k": 480,
    "efficientnetv2-xl-21k-ft1k": 512,
    "efficientnetv2-b0-21k-ft1k": 224,
    "efficientnetv2-b1-21k-ft1k": 240,
    "efficientnetv2-b2-21k-ft1k": 260,
    "efficientnetv2-b3-21k-ft1k": 300, 
    "efficientnet_b0": 224,
    "efficientnet_b1": 240,
    "efficientnet_b2": 260,
    "efficientnet_b3": 300,
    "efficientnet_b4": 380,
    "efficientnet_b5": 456,
    "efficientnet_b6": 528,
    "efficientnet_b7": 600,
    "inception_v3": 299,
    "inception_resnet_v2": 299,
    "nasnet_large": 331,
    "pnasnet_large": 331,
    }

    #-------------- Selectbox and buttons --------------#
    df = pd.DataFrame(list(model_image_size_map.items()), columns=['Model', 'img size'])

    col1, col2 = st.columns([1, 1])

    col1.subheader("Select model")
    selected_model_name = col1.selectbox(
            'Select the model to train and evaluate:',
            model_image_size_map.keys()
            )

    model_handle = model_handle_map.get(selected_model_name)
    pixels = model_image_size_map.get(selected_model_name, 224)
    IMAGE_SIZE = (pixels, pixels)

    col1.markdown("""
        Selected model : {temp1}  
        model : {temp2}  
        img size: {temp3} px"""
            .format(temp1=selected_model_name, 
                    temp2=model_handle,
                    temp3=IMAGE_SIZE)
    )
    
    col2.subheader("Model info")
    col2.dataframe(df)
    
    
    #-------------- Display pretrained model history --------------#
    if st.button("Next step", type="primary"):
        
        # load and display model history
        st.write(' - Model train history:')
        st
        model_history_path = f"./saved_models_hist/saved_mushrooms_model_hist_{selected_model_name}.npy"
        hist = np.load(model_history_path,allow_pickle='TRUE').item()

        fig = plt.figure(figsize=(12,4))

        plt.subplot(121)
        plt.title(selected_model_name + ' loss by epoch')
        plt.ylabel("Loss (training and validation)")
        plt.xlabel("Training Steps")
        plt.ylim([0,2])
        plt.plot(hist["loss"])
        plt.plot(hist["val_loss"])
        plt.legend(['train', 'val'], loc='right')

        plt.subplot(122)
        plt.title(selected_model_name + ' acc by epoch')
        plt.ylabel("Accuracy (training and validation)")
        plt.xlabel("Training Steps")
        plt.ylim([0,1])
        plt.plot(hist["accuracy"])
        plt.plot(hist["val_accuracy"])
        plt.legend(['train', 'val'], loc='right')
        
        st.pyplot(fig)

        # ------------ load and resize test image dataset ---------------
        img_dir = unzip_images()
        st.write(' - Test images file unzipped')

        normalization_layer = tf.keras.layers.Rescaling(1. / 255)

        test_ds = build_dataset(img_dir, IMAGE_SIZE)
        class_names = tuple(test_ds.class_names)
        test_size = test_ds.cardinality().numpy()
        test_ds = test_ds.unbatch().batch(16)
        test_ds = test_ds.map(lambda images, labels:
                    (normalization_layer(images), labels))
        
        # enumerate unzipped files
        # Trouver tous les chemins vers les fichiers qui finissent par .jpg
        #liste = glob.glob(f"{img_dir}/*/*.jpg")

        # The map() function executes a specified function for each item in an iterable.
        # Remplacer les \\ par /
        #liste = list(map(lambda x : x.replace('\\','/'), liste))
        # Extraire la classe de champignons du path dans une liste à 3 colonnes: [file path, File name, Class]
        #img_test_list = list(map(lambda x : [x, x.split('/')[4], x.split('/')[3]], liste))
        #df_test_img = pd.DataFrame(img_test_list, columns=['filepath', 'filename', 'nameLabel'])
        #df_test_img['label'] = df_test_img['nameLabel'].replace(df_test_img.nameLabel.unique(), [*range(len(df_test_img.nameLabel.unique()))])
        # This code could be simplified (refactoring)

        #X_test_path, y_test_label = df_test_img.filepath, df_test_img.label

        # Charger les images de Validation (X_test_path) redimensionnées à [224,224,3] en mémoire dans la variable X_test.
        #X_test = []
        #for filepath in X_test_path:
        #    # Lecture du fichier
        #    im = tf.io.read_file(filepath)
        #    # On décode le fichier
        #    im = tf.image.decode_jpeg(im, channels=3)
        #    # Redimensionnement
        #    im = tf.image.resize(im, size=(pixels, pixels))
        #    X_test.append([im])

        #X_test = tf.concat(X_test, axis=0)

        #-------------- Load pretrained model, predict on test image dataset --------------#
        #model = tf.saved_model.load(f"./saved_models/saved_mushrooms_model_{selected_model_name}")
        # The object returned by tf.saved_model.load is not a Keras object (i.e. doesn't have .fit, .predict, etc. methods)
        st.write(' - Loading trained model...')
        do_fine_tuning = False
        # reload custom model: https://www.tensorflow.org/tutorials/keras/save_and_load#saving_custom_objects
        model = tf.keras.models.load_model(f"./saved_models/saved_mushrooms_model_{selected_model_name}.keras",
                                            custom_objects={'KerasLayer': hub.KerasLayer(model_handle, trainable=do_fine_tuning)})
        st.write(' - Restored model summary:')
        model.summary(print_fn=lambda x: st.text(x))

        # Evaluate the restored model
        st.write(' - Evaluating restored model on test img dataset...')
        loss, acc = model.evaluate(test_ds, verbose=2)
        st.write('Accuracy: {:5.2f}%'.format(100 * acc))
        st.write('Loss: {:5.2f}'.format(loss))
        #print(model.predict(test_images).shape)
        y_prob = model.predict(test_ds, batch_size=16)
        # Prédiction de la classe
        y_pred = tf.argmax(y_prob, axis=-1).numpy()

        # ------------ Affichage des images avec la prédiction ---------------
        number_of_images = 6 # définir le nombre x d'images à afficher 
        #indices_random = tf.random.uniform([number_of_images], 0, len(test_ds), dtype=tf.int32)
        dsit = iter(test_ds)

        fig = plt.figure(figsize=(15,15))
        for i in range(0,number_of_images):
            x, y = next(dsit)
            image = x[0, :, :, :]
            true_index = np.argmax(y[0])

            # Expand the validation image to (1, 224, 224, 3) before predicting the label
            prediction_scores = model.predict(np.expand_dims(image, axis=0))
            predicted_index = np.argmax(prediction_scores)

            plt.subplot(int(number_of_images/2),3,i+1)
            plt.imshow(image)
            plt.axis('off')
            plt.title('Pred class : {} \n Real class : {}'.format(class_names[predicted_index], class_names[true_index]))

        st.pyplot(fig)
        


