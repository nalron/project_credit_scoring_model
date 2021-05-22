# Projet : Implémentez un modèle de scoring
### Projet du parcours Data Scientist OpenClassrooms en partenariat avec Centrale Supélec.
L’entreprise souhaite développer un **modèle de scoring de la probabilité de défaut de paiement du client** pour étayer la décision d'accorder ou non un prêt à un client potentiel en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).


## Dashboard Scoring Credit
[Web App from Heroku](https://bank-credit-risk.herokuapp.com/) & [Sharing Streamlit](https://share.streamlit.io/nalron/project_credit_scoring_model/app/app.py)


L'application répond au **cahier des charges** suivant :
 - Permettre de visualiser le score et l’interprétation de ce score pour chaque client pour une personne non experte en data science.
 - Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre).
 - Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires.


**Ngrok** a été utilisé pour le développement de l'application en local. Ngrok est un outil qui permet de rendre votre poste de développement web local accessible depuis l'extérieur, via un tunnel sécurisé, depuis une url du type https://azerty.ngrok.io.

```
pip install pyngrok==4.1.1
```

## Préambule
Pour ce projet, les données ont été manipulées en Python sur support Jupyter Notebook avec développement de l'app Streamlit. Google Colab a été utilisé pour l'entraînement du modèle en exécution GPU.

### Les données
 - Données Kaggle : [Home Credit Default](https://www.kaggle.com/c/home-credit-default-risk/data)

### Compétences évaluées
 - Présenter son travail de modélisation à l'oral
 - Déployer un modèle via une API dans le Web
 - Utiliser un logiciel de version de code pour assurer l’intégration du modèle
 - Rédiger une note méthodologique afin de communiquer sa démarche de modélisation
 - Réaliser un dashboard pour présenter son travail de modélisation

## Prérequis techniques
Si vous n'avez jamais installé **Python**, alors autant installer directement la **distribution Anaconda**.
Anaconda est donc une distribution Python, faite pour la Data Science.

De cette manière on peut installer Python et ses librairies de Data Science Pandas, Matplotlib, Seaborn, Scipy, Numpy etc… 
Mais aussi le notebook Jupyter, qui reste incontournable et vivement recommandé!
C'est par ici : [Anaconda](https://www.anaconda.com/download)

Si vous souhaitez lancer le projet, il sera nécessaire d'installer Jupyter Notebook sur votre mahcine. 
La doc. Jupyter est accessible via : [Jupyter Documentation](https://jupyter.readthedocs.io/en/latest/install.html) 

```
python -m pip install --upgrade pip    
python -m pip install jupyter
```

Pour tester l'installation, vous pouvez taper dans votre console la commande suivante :

```
jupyter notebook
```

### Installation des principales librairies Python uniquement
*Pour installer python ainsi que les librairies de Data Science, il est fortement recommandé d'installer la distribution Anaconda.* 

```
pip install pandas
pip install matplotlib
pip install numpy
pip install scipy
```

### Heroku Git
*Installation de Heroku Command Line Interface (CLI) [Heroku CLI](https://devcenter.heroku.com/articles/heroku-command-line)* 

```
$ heroku login
```
Mise à jour des modifications
```
$ git add .
$ git commit -am "make it better"
$ git push heroku master
```

### Ressources Streamlit [Cheat Sheet…](https://share.streamlit.io/daniellewisdl/streamlit-cheat-sheet/app.py)

## Auteur

**Nalron** *(Nicolas Pasero)* [Persona](https://nalron.com) - *Initial work* - [Github](https://github.com/nalron)
