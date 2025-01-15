README : Visualisation d'un arbre de décision pour Titanic

Description

Ce projet consiste à générer un arbre de décision pour analyser le jeu de données Titanic et prédire la survie en fonction de certaines caractéristiques telles que la classe du passager (Pclass), l'âge (Age) et le sexe (Sex). Le projet utilise Python pour le traitement des données, l'entraînement du modèle et la visualisation.

Prérequis

Assurez-vous que les bibliothèques Python suivantes sont installées :

pandas : Pour la manipulation des données.

scikit-learn : Pour créer et visualiser l'arbre de décision.

matplotlib : Pour tracer l'arbre de décision.

pymysql (optionnel) : Pour l'interaction directe avec la base de données (si vous utilisez Navicat).

Installez ces bibliothèques avec pip :

pip install pandas scikit-learn matplotlib pymysql

Jeu de données

Le jeu de données doit inclure au minimum les colonnes suivantes :

Pclass : Classe du passager (1ère, 2ème ou 3ème).

Age : Âge du passager.

Sex : Sexe du passager (male/female).

Survived : Statut de survie (0 = Non survécu, 1 = Survécu).

Si vous utilisez un fichier CSV, assurez-vous qu'il est correctement formaté et qu'il ne contient pas de valeurs manquantes dans les colonnes nécessaires pour l'analyse.

Étapes d'exécution

1. Charger les données

Utilisez le jeu de données Titanic fourni (titanic.csv) ou connectez-vous à votre base de données (par exemple, Navicat) pour récupérer les données.

import pandas as pd

# Charger les données depuis un fichier CSV
data = pd.read_csv('titanic.csv')

2. Prétraiter les données

Gérez les valeurs manquantes.

Encodez les variables catégoriques (par exemple, Sex et Embarked).

from sklearn import preprocessing

data = data.dropna()
data['Sex'] = preprocessing.LabelEncoder().fit_transform(data['Sex'])

3. Entraîner l'arbre de décision

Sélectionnez les caractéristiques (Pclass, Age, Sex) et la variable cible (Survived).

Divisez les données en ensembles d'entraînement et de test.

Entraînez le DecisionTreeClassifier.

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

features = ['Pclass', 'Age', 'Sex']
X = data[features]
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

4. Visualiser l'arbre de décision

Utilisez matplotlib et sklearn.tree.plot_tree pour générer la visualisation de l'arbre de décision.

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(15, 10))
plot_tree(
    clf,
    feature_names=features,
    class_names=['Not Survived', 'Survived'],
    filled=True,
    rounded=True
)
plt.show()

5. Optionnel : Connexion à la base de données

Si vous souhaitez charger les données directement depuis une base de données (par exemple, MySQL/Navicat), utilisez la bibliothèque pymysql :

import pymysql

connection = pymysql.connect(
    host='localhost',
    user='votre_utilisateur',
    password='votre_mot_de_passe',
    database='votre_base_de_données'
)
query = "SELECT Pclass, Age, Sex, Survived FROM titanic_table WHERE Age IS NOT NULL"
data = pd.read_sql(query, connection)
connection.close()

Résultat

La sortie est une représentation visuelle de l'arbre de décision, qui fournit des informations sur les règles utilisées par le modèle pour prédire la survie. Chaque nœud de l'arbre divise le jeu de données en fonction des seuils des caractéristiques, et les feuilles indiquent les prédictions finales.
