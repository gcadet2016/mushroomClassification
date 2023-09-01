# Jul23_bds_champignons



# Pour les autres contributeurs

1- Commencer par cloner la repository localement  
Cette opération n'est à faire qu'**une seule fois** lorsque vous n'avez jamais récupéré sur votre PC (ou autre) le projet de Github.


- Aller sur la page https://github.com/DataScientest-Studio/Jul23_bds_champignons  
- Clicker sur le bouton  
![Button](img\button.png)
- Copier l'url  
![Urlcopy](img\url_copy.png)


- Ouvrir un invite de commande powershell  
- Executer les commandes suivantes (à adapter selon votre besoin)  
Le plus important est la commande **git clone**
![GithubClone](img\GithubClone.png) 

Par la suite vous devrez faire un "git pull" pour mettre à jour votre projet avec les nouveautés que vos collaborateurs ont poussé sur Github.

## La suite de ce tutoriel est écrit pour VSCode.  
- L'usage de Git dans VsCode est simple.  
- L'usage de Git en lignes de commandes est plus complexe et constitue une source d'erreur pour les utilisateurs avec peu d'expérience.

De plus vous serez très certainement amenés à l'avenir à utiliser VsCode pour vos projets.

Donc si vous travaillez avec un autre environnement comme Anaconda, il suffit de sauvegarder les fichiers dans le dossier local du projet Github puis de pousser l'ensemble vers Github avec VsCode ou des commandes git.

### 2- [Télécharger et Installer VsCode](https://code.visualstudio.com/download)  


### 3- Créer votre branche de travail  
Recommandation: mettre son alias dans le nom de la branche afin qu'on sache à qui elle appartient:

Par défaut vous êtes dans la branche "main".  
- Cliquer sur la branche main  
![Alt text](img\mainBranch.png)  
Le menu de commande s'ouvre en haut de VsCode:  
![Alt text](img\newBranch1.png)  
- Cliquer sur **Create new branch...**  
- Saisir le nom de la nouvelle branche. Exemple "Jacky_myFirstNotebook".  
Désormais vous êtes dans votre branche de travail qui est a été clonéé à partir de la branche "main".
Vous pouvez commencer à travailler et faire vos changements.

### 4- Sauvegarder tous vos fichiers avant de pousser votre branche vers GitHub
4- Faire un commit qui va inclure vos changements.  
Remarque: vous pouvez faire plusieurs changements, plusieurs commits, l'important étant que la dernière opération doit être un commit avant de pousser votre branche dans GitHub.  
![Alt text](img\commit.png)

- **Saisir un court message** résumant en quoi consiste le changement
- Vérifier la liste des fichiers que vous venez de modifier (optionel).
- Cliquer sur le bouton **Commit**  

4- Poussez votre branche vers Github: cliquer sur **Sync Changes**  
![Alt text](img\syncBranch.png)  

5- Vérifier sur Github (optionel)
- Afficher la liste des branches
- Sélectionnez votre branche  
![Alt text](img\selectBranch.png)  
- Accédez à vos fichiers
![Alt text](img\branch.png)

6- En cours de développement, vous pouvez vérifier le status de votre branche:  
ici  
![Alt text](img\status1.png)  
Le chiffre indique le nombre de changement en attente de commit  
et ici  
![Branch status](img\status2.png)  
L'étoile signifie que votre branche local devrait être poussée vers GitHub car il y a des changements en local non répliqués sur Github.