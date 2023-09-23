# Version 1.4

# -------------------------------------------------------------------------------------------------
# Fonction de création du generator
# -------------------------------------------------------------------------------------------------
def flow_datagen(datagen, train_val_test, img_dim, batch_size):

    generator = datagen.flow_from_dataframe(
        dataframe = train_val_test,
        x_col='image_url',               # Colonne contenant les chemins des images
        y_col='label',                   # Colonne contenant les étiquettes (classes)
        target_size = img_dim,           # Taille cible des images
        batch_size = batch_size,         # Taille du lot
        class_mode='categorical',        # Mode de classification
        shuffle= True)                   # Mélanger les données
    
    return generator