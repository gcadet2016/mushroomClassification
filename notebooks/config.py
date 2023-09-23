# v1.4
    # Ajout de config_etude()

# v1.3
    # Amélioration de la prise en charge de l'environnement collab à travers les chemins d'accès définis

# v1.2
    # Ajout boucle while pour assurer le bon choix des variables

# v1.1
    # améliorations code

# v1.0
    # Construction fichier



def config_env():


    # Nom du modèle
    model_name = input('Préciser nom à donner au modèle:')



    # Choix fichier de données :
    while True:
        print("\n Preciser le fichier de données à utiliser :")
        print("1. 10 classes, 64k images, non triées")
        print("2. 10 classes, 60k images, triées")
        data_file = input("\n Entrez le numéro correspondant au fichier de données choisi : ")

        if data_file == "1":
            print("Selection : 10 classes, 64k images, non triées")
            data_file = '/top10_no_tri.csv'
            data_name = '_no_tri'
            break
    
        elif data_file == "2":
            print("Selection : 10 classes, 60k images, triées")
            data_file = '/top10.csv'
            data_name = '_tri'
            break

        else:
            print("Choix non valide")


    
    # Choix tailles des données :
    while True:
        print("\n L'execution porte sur les données complètes ou un echantillon aléatoire des données ?:")
        print("1. Donnée complètes")
        print("2. Echantillon")
        data_size = input("\n Entrez le numéro correspondant au choix: ")

        if data_size == "1":
            print("Selection : Toutes les données")
            pourcentage_echantillon = 0.1
            data_choice = '_full'
            break

        elif data_size == "2":
            print("Selection : Echantillon")
            print("Preciser la taille de l'echantillon, 0.1 correspondant à 10% des données")
            pourcentage_echantillon = input("Entrer la taille de l'echantillon souhaité : (0.1 = 10%)")
            pourcentage_echantillon = float(pourcentage_echantillon)
            print('Taille echantillon :', pourcentage_echantillon*100,'%')
            data_choice = '_ech'
            break


        else:
            print("Choix non valide")


    # Choix undersampling :
    while True:
        print("\n Les données doivent-elles équilibrées ? (un undersampling sera executé):")
        print("1. AVEC undersampling")
        print("2. SANS undersampling")
        ans_sampling = input("\n Entrez le numéro correspondant au choix: ")

        if ans_sampling == "1":
            print("Selection : Avec undersampling")
            undersampling = True
            sample_choice = '_sample'
            break

        elif ans_sampling == "2":
            print("Selection : Sans undersampling")
            undersampling = False
            sample_choice = '_no_sample'
            break

        else:
            print("Choix non valide")


    # Choix generateur :
    while True:
        print("\n Preciser generateur d'images à utiliser:")
        print("1. Keras generator")
        print("2. Tensorflow Dataset (K.O)")
        generator_choice = input("\n Entrez le numéro correspondant au choix: ")

        if generator_choice == "1":
            print("Keras generator")
            gen_choice = '_keras_generator'
            break

        elif generator_choice == "2":
            print("Selection : Tensorflow Dataset (K.O)")
            gen_choice = '_tensorflow_dataset'
            break

        else:
            print("Choix non valide")



    # Environnement de travail :
    while True:
        print("\n Preciser l'environnement de travail :")
        print("1. Ma machine")
        print("2. Google Colab")
        work_env = input("\n Entrez le numéro correspondant à l'environnement choisi : ")

        if work_env == "1":
            print("Environnement : Perso")
            chemin_images = input("Préciser l'URL du dossier image se trouvant sur votre machine:" \
                                    "\n Ex :'../../images'")
            print('Chemin des images:', chemin_images)
            data_path = '../data'
            history_path = '../history/'
            model_path = '../model/'
            break
    
        elif work_env == "2":
            print("Environnement : Google Colab \n Les images seront decompressées ici : '/images/images'")
            chemin_images = '/images/images'
            data_path = '/content/drive/MyDrive/SAS/Jul23_bds_champignons/data'
            history_path = '/content/drive/MyDrive/SAS/history/'
            model_path = '/content/drive/MyDrive/SAS/model/'
            break

        else:
            print("Choix non valide")


    data_path = data_path + data_file
    print('Chemin du fichier de données :', data_path)

    model_name = model_name + data_name + data_choice + sample_choice + gen_choice
    print('Nom du modele defini :', model_name)

    model_path = model_path + model_name
    print('Chemin d\'accès au modèle enregistré:', model_path)

    history_path = history_path + model_name
    print('Chemin d\'accès à l\'historique d\'entrainement enregistré:', history_path)


    return model_name, history_path, model_path, data_path, data_size, chemin_images, pourcentage_echantillon, undersampling, generator_choice, work_env


def config_etude():

    # Choix modèle à importer :
    while True:
        print("\n Preciser le modèle à importer:")
        print("1. V1")
        print("2. V2")
        print("1. V3 (non prête)")
        print("2. V4 (non prête)")
        print("2. V5 (non prête)")  
        modele_choice = input("\n Entrez le numéro correspondant au modèle choisi : ")

        if modele_choice == "1":
            print("Selection : V1 \n")
            print("Modèle itération V1 : \n"\
                  "Modèle pré-entrainé : EfficientNetv2B0 21k \n"\
                  "Générateur : Keras ImageGenerator \n"\
                  "Images : Triées \n"\
                  "Undersample : Oui \n"\
                  "Set de données : ech 10% \n")
                  
            model_choice = '../model/V1'
            history_path = '../history/V1.pkl'
            generator_choice = 'keras'
            img_dim = (224,224)
            img_shape = (224,224,3)
            batch_size = 64
            break


        elif modele_choice == "2":
            print("Selection : V2 \n")
            print("Modèle itération V2 : \n"\
                  "Modèle pré-entrainé : EfficientNetv2B0 21k \n"\
                  "Générateur : Keras ImageGenerator \n"\
                  "Images : Triées \n"\
                  "Undersample : Oui \n"\
                  "Set de données : full \n")
                  
            model_choice = '../model/V2'
            history_path = '../history/V2.pkl'
            generator_choice = 'keras'
            img_dim = (224,224)
            img_shape = (224,224,3)
            batch_size = 64
            break

    

        else:
            print("Choix non valide - Modèle non prêt")


    return model_choice, generator_choice, history_path, img_dim, img_shape, batch_size


