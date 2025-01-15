# 📘 README : Visualisation d'un arbre de décision pour Titanic

## 🛠️ Description
Ce projet consiste à générer un arbre de décision pour analyser le jeu de données Titanic et prédire la survie en fonction de certaines caractéristiques telles que :
- **Classe du passager** (`Pclass`)
- **Âge** (`Age`)
- **Sexe** (`Sex`)

Le projet utilise Python pour le traitement des données, l'entraînement du modèle et la visualisation.

---

## 📋 Prérequis
### Assurez-vous que les bibliothèques suivantes sont installées :
- **`pandas`** : Manipulation des données.
- **`scikit-learn`** : Création et visualisation de l'arbre de décision.
- **`matplotlib`** : Visualisation de l'arbre de décision.
- **`pymysql`** *(optionnel)* : Interaction avec une base de données (Navicat, MySQL).

#### Commande pour les installer :
```bash
pip install pandas scikit-learn matplotlib pymysql
```

---

## 🗂️ Jeu de données
### Le dataset doit contenir les colonnes suivantes :
| Colonne     | Description                          |
|-------------|--------------------------------------|
| **Pclass**  | Classe du passager (1ère, 2ème ou 3ème) |
| **Age**     | Âge du passager                    |
| **Sex**     | Sexe du passager (`male` / `female`) |
| **Survived**| Statut de survie (0 = Non, 1 = Oui)  |


---

## ⚙️ Étapes d'exécution

### **1. Charger les données**
- Utilisez un fichier CSV c `titanic.csv` 

```python
import pandas as pd

# Charger les données depuis un fichier CSV
data = pd.read_csv('titanic.csv')
```

---

### **2. Prétraiter les données**
- Gérez les valeurs manquantes.
- Encodez les variables catégoriques (par exemple, `Sex`).

```python
from sklearn import preprocessing

data = data.dropna()  # Supprimer les lignes avec des valeurs manquantes
data['Sex'] = preprocessing.LabelEncoder().fit_transform(data['Sex'])
```

---

### **3. Entraîner l'arbre de décision**
- Sélectionnez les caractéristiques et la variable cible.
- Divisez les données en ensembles d'entraînement et de test.
- Entraînez le modèle.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

features = ['Pclass', 'Age', 'Sex']
X = data[features]
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)
```

---

### **4. Visualiser l'arbre de décision**
- Utilisez `matplotlib` pour afficher l'arbre.

```python
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
```

---

### **5. Optionnel : Connexion à une base de données**
Si vous souhaitez charger les données directement depuis une base de données (par exemple, MySQL/Navicat), utilisez `pymysql` :

```python
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
```

---

## 📊 Résultat
- **Sortie** : Une représentation visuelle de l'arbre de décision.
- **Noeuds** : Divisions basées sur les caractéristiques importantes (e.g., Sexe, Classe).
- **Feuilles** : Prédictions finales (Survived ou Not Survived).

### Exemple :
```text
                          [Sex <= 0.5]
                          /         \
                [Age <= 3.0]      [Pclass <= 2.5]
               /       \           /          \
          [Survived] [Not Survived] [Survived]
```

---

## ⚠️ Limitations
- Toutes les caractéristiques doivent être numériques ou encodées.
- Les valeurs manquantes doivent être gérées avant l'entraînement.
- La profondeur de l'arbre est limitée à `max_depth=4` pour éviter le surapprentissage.

---




