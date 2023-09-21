
def config_env():


    # Nom du modèle
    model_name = input('Préciser nom à donner au modèle:')

    # Choix fichier de données :
    print("\n Preciser le fichier de données à utiliser :")
    print("1. 10 classes, 64k images, non triées")
    print("2. 10 classes, 60k images, triées")
    data_file = input("\n Entrez le numéro correspondant au fichier de données choisi : ")

    if data_file == "1":
        print("Selection : 10 classes, 64k images, non triées")
        data_file = '../data/top10_no_tri.csv'
        data_name = '_no_tri'
    
    elif data_file == "2":
        print("Selection : 10 classes, 60k images, triées")
        data_file = '../data/top10.csv'
        data_name = '_tri'

    else:
        print("Choix non valide")


    # Choix tailles des données :
    print("\n L'execution porte sur les données complètes ou un echantillon aléatoire des données ?:")
    print("1. Donnée complètes")
    print("2. Echantillon")
    data_size = input("\n Entrez le numéro correspondant au choix: ")

    if data_size == "1":
        print("Selection : Toutes les données")
        pourcentage_echantillon = 0.1

    elif data_size == "2":
        print("Selection : Echantillon")
        print("Preciser la taille de l'echantillon, 0.1 correspondant à 10% des données")
        pourcentage_echantillon = input("Entrer la taille de l'echantillon souhaité : (0.1 = 10%)")
        pourcentage_echantillon = float(pourcentage_echantillon)
        print('Taille echantillon :', pourcentage_echantillon*100,'%')


    else:
        print("Choix non valide")


    # Choix undersampling :
    print("\n Les données doivent-elles équilibrées ? (un undersampling sera executé):")
    print("1. AVEC undersampling")
    print("2. SANS undersampling")
    ans_sampling = input("\n Entrez le numéro correspondant au choix: ")

    if ans_sampling == "1":
        print("Selection : Avec undersampling")
        undersampling = True

    elif ans_sampling == "2":
        print("Selection : Sans undersampling")
        undersampling = False

    else:
        print("Choix non valide")



    # Environnement de travail :
    print("\n Preciser l'environnement de travail :")
    print("1. Ma machine")
    print("2. Google Colab")
    work_env = input("\n Entrez le numéro correspondant à l'environnement choisi : ")

    if work_env == "1":
        print("Environnement : Perso")
        chemin_images = input("Préciser l'URL du dossier image se trouvant sur votre machine:" \
                             "\n Ex :'../../images'")
        print('Chemin des images:', chemin_images)
        model_name = '../model/' + model_name
        history_path = '../history/history_' + model_name
    
    elif work_env == "2":
        print("Environnement : Google Colab")
        chemin_images = '/images/images/'
        data_file = '/content/drive/MyDrive/SAS/Jul23_bds_champignons/data/top10.csv'
        model_name = '/content/drive/MyDrive/SAS/model/' + model_name
        history_path = '/content/drive/MyDrive/SAS/history/' + model_name
    
    else:
        print("Choix non valide")

    # Dimensions des images à traiter
    img_dim = (200,200)
    img_shape = (200,200,3)

    return work_env, chemin_images, data_file, data_name, data_size, pourcentage_echantillon, undersampling, model_name, history_path, img_dim, img_shape