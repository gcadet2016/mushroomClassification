import streamlit as st
import itertools
import os

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import zipfile

import tensorflow as tf
import tensorflow_hub as hub

title = "Test a model"
sidebar_name = "Test a model"

# All path and file names are hard coded for now
def unzip_images():
    zip_file = './assets/imgSample.zip'
    with zipfile.ZipFile(zip_file,"r") as z:
        z.extractall("./assets")
    return "./assets/imgSample"

def build_dataset(subset, data_dir, IMAGE_SIZE):
  return tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=.20,
      subset=subset,
      label_mode="categorical",
      # Seed needs to provided when using validation_split and shuffle = True.
      # A fixed seed is used so that the validation set is stable across runs.
      seed=123,
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
    
    
    #-------------- Prepare datasets, create model, train model... --------------#
    if st.button("Next step", type="primary"):
        
        BATCH_SIZE = 16
        st.write('Preparing datasets...')
        img_dir = unzip_images()
        st.write(' - img files unzipped')

        train_ds = build_dataset("training", img_dir, IMAGE_SIZE)
        class_names = tuple(train_ds.class_names)
        train_size = train_ds.cardinality().numpy()
        train_ds = train_ds.unbatch().batch(BATCH_SIZE)
        train_ds = train_ds.repeat()
        
        normalization_layer = tf.keras.layers.Rescaling(1. / 255)
        preprocessing_model = tf.keras.Sequential([normalization_layer])
        st.write(' - normalization done')

        do_data_augmentation = False
        if do_data_augmentation:
            preprocessing_model.add(
                tf.keras.layers.RandomRotation(40))
            preprocessing_model.add(
                tf.keras.layers.RandomTranslation(0, 0.2))
            preprocessing_model.add(
                tf.keras.layers.RandomTranslation(0.2, 0))
            # Like the old tf.keras.preprocessing.image.ImageDataGenerator(),
            # image sizes are fixed when reading, and then a random zoom is applied.
            # If all training inputs are larger than image_size, one could also use
            # RandomCrop with a batch size of 1 and rebatch later.
            preprocessing_model.add(
                tf.keras.layers.RandomZoom(0.2, 0.2))
            preprocessing_model.add(
                tf.keras.layers.RandomFlip(mode="horizontal"))
            st.write(' - data augmentation done')
        else:
            st.write(' - data augmentation disabled')

        train_ds = train_ds.map(lambda images, labels:
                        (preprocessing_model(images), labels))
        st.write(' - train dataset created')

        val_ds = build_dataset("validation", img_dir, IMAGE_SIZE)
        valid_size = val_ds.cardinality().numpy()
        val_ds = val_ds.unbatch().batch(BATCH_SIZE)
        val_ds = val_ds.map(lambda images, labels:
                    (normalization_layer(images), labels))
        st.write(' - validation dataset created')

        # Defining model
        do_fine_tuning = False
        if do_fine_tuning:
            st.write(" - Building model with fine-tuning")
            pass
        else:
            st.write(" - Building model without fine-tuning")
            
            model = tf.keras.Sequential([
                # Explicitly define the input shape so the model can be properly
                # loaded by the TFLiteConverter
                tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
                hub.KerasLayer(model_handle, trainable=do_fine_tuning),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(len(class_names),
                                    kernel_regularizer=tf.keras.regularizers.l2(0.0001))
            ])
            model.build((None,)+IMAGE_SIZE+(3,))

        model.summary(print_fn=lambda x: st.text(x))
    
        # Training model
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9), 
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
            metrics=['accuracy']
            )
        st.write(" - model compiled")

        st.write(" - model training started...")
        steps_per_epoch = train_size // BATCH_SIZE
        validation_steps = valid_size // BATCH_SIZE
        hist = model.fit(
            train_ds,
            epochs=5, steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=validation_steps).history(print_fn=lambda x: st.text(x))

# pas trouvé de solution pour sortir le résultat de l'entrainement


