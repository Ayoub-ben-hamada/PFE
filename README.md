# SADP - Volet AI & Data Science

## 1) Présentation générale du projet SADP
Ce dépôt implémente le volet AI & Data Science du projet SADP (Système d'Aide à la Décision et de Distribution des Produits), centré sur l'analyse d'installations de produits logiciels et la modélisation du risque d'installation.

Le travail couvre:
- La préparation et le nettoyage de données d'installation.
- L'analyse exploratoire statistique (EDA) et la construction de KPI.
- La construction de modèles de classification et de régression.
- Une couche d'interprétabilité (SHAP) et de recommandations opérationnelles dans la version SAD.

## 2) Problématique et objectifs
Le pipeline vise à transformer des données brutes d'installation en informations décisionnelles exploitables:
- Qualifier la réussite d'installation et la satisfaction.
- Identifier les clients/situations à risque.
- Estimer la durée d'installation.
- Expliquer les facteurs de risque et proposer des actions prioritaires.

## 3) Architecture et organisation du dépôt
Arborescence réelle du dépôt:

```text
.
├── analyse.ipynb
├── analyse.py
├── analysecopy.py
├── analyse_copy.ipynb
├── analyse_copy.py
├── base_de_donnees_installations_produits_CNI.csv
├── base_de_données_installations_produits_CNI_4.csv
├── base_finale_v3.csv
├── base_finale_v4.csv
├── installations_produits_CNI.csv
├── installations_produits_CNI_clean.csv
├── SAD.ipynb
└── .vscode
    └── settings.json
```

## 4) Description détaillée des différents modules
| Fichier | Rôle fonctionnel réel |
|---|---|
| analyse.py | Script exporté de notebook: EDA + préparation + feature engineering + modèles classification/régression. Contient aussi des blocs commentés (génération synthétique, équilibrage avancé). |
| analysecopy.py | Variante de pipeline (basée sur base_finale_v3.csv): EDA, KPI, 5 cibles de modélisation (4 classification + 1 régression), synthèse des modèles. |
| analyse_copy.py | Version la plus complète: nettoyage avancé de Remarques, export d'un CSV propre, KPI, modélisation, fine-tuning (GridSearchCV), SHAP, génération de recommandations pour un usage SAD. |
| analyse.ipynb / analyse_copy.ipynb / SAD.ipynb | Versions notebook des pipelines (exploration, visualisation, modélisation). SAD.ipynb reprend la logique orientée aide à la décision. |
| fichiers CSV | Jeux de données à différents stades (brut, versions intermédiaires/finales, nettoyé). |

Remarque importante:
- Les scripts .py proviennent d'exports notebook et incluent des variantes/historiques de pipeline.
- Le fichier analyse_copy.py est le plus représentatif d'un flux complet de bout en bout avec explicabilité.

## 5) Prétraitement et préparation des données
Méthodes effectivement implémentées:

### 5.1 Corrections de type et normalisation
- Conversion de date_installation en datetime.
- Conversion de Debit_reseau_Mbps en float (remplacement de virgule).
- Conversion de version_produit selon les besoins (object/numérique selon étapes).
- Harmonisation de modalités incohérentes (ex: win10 / Win 10 / Win10).

### 5.2 Déduplication et filtrage
- Suppression des doublons.
- Détection et traitement des valeurs aberrantes (IQR) sur certaines variables.
- Suppression explicite de valeurs incohérentes (ex: nombre_reclamations_client == -2).

### 5.3 Gestion des valeurs manquantes
- Règle de seuil: suppression des lignes pour colonnes avec taux de NA faible.
- Imputation de Remarques via règles métier/statistiques:
  - Règles conditionnelles sur produit, taille site, version, mode d'installation, gouvernorat, OS, débit réseau.
  - Affectation de Aucune pour certaines combinaisons réussies/satisfaites.

### 5.4 Sortie de données nettoyées
- Export explicite dans analyse_copy.py:
  - installations_produits_CNI_clean.csv

## 6) Analyse exploratoire et statistiques descriptives
Analyses implémentées:
- Statistiques descriptives (variables numériques et qualitatives).
- Distribution des variables (histogrammes, KDE, boxplots).
- Matrice de covariance et matrice de corrélation.
- Tableaux croisés par catégories (produit, type client, gouvernorat, mode d'installation, OS, etc.).
- KPI globaux et KPI par segment (produit, version, client, zone).
- Visualisations statiques (Matplotlib/Seaborn) et interactives (Plotly).

Tests/statistiques utilisés:
- Khi-deux (chi2_contingency) pour dépendances catégorielles.
- Information mutuelle (mutual_info_classif) pour l'analyse de dépendance entre variables et Remarques.

## 7) Détection et analyse des anomalies
Ce dépôt met en oeuvre principalement une détection d'anomalies orientée règles et statistiques descriptives:
- Incohérences de catégories (ex: variantes d'OS).
- Valeurs invalides (ex: réclamations négatives).
- Détection par IQR (outliers numériques).
- Contrôles temporels (dates aberrantes).

Ce qui n'est pas implémenté:
- Pas d'algorithmes dédiés d'anomaly detection de type IsolationForest, OneClassSVM, LocalOutlierFactor.

## 8) Méthodes statistiques et algorithmes de Machine Learning
### 8.1 Encodage et transformation
- Encodage binaire (oui/non -> 1/0) pour plusieurs cibles.
- One-hot encoding pour catégories (produit, type client, mode, catégories regroupées).
- Target encoding (category_encoders.TargetEncoder) sur certaines variables (notamment OS et version) dans les blocs de modélisation.

### 8.2 Feature engineering
Variables construites:
- Catégorisation taille site: cat_taille.
- Catégorisation débit réseau: cat_debit.
- Zone d'urbanisation depuis gouvernorat.
- Variables temporelles: annee, jour_semaine, mois, mois_num, weekend selon script.
- Variables issues de Remarques: pb_reseau, pb_configuration, pb_materiel, pb_version, pb_aucun.
- Variables de risque: score_risque puis client_risque.

### 8.3 Sélection de variables
- Mutual Information (classification/régression) avec seuil + top-k.

### 8.4 Modèles de classification utilisés
- LogisticRegression
- DecisionTreeClassifier
- RandomForestClassifier
- GradientBoostingClassifier
- XGBClassifier (XGBoost)
- KNeighborsClassifier
- SVC

### 8.5 Modèles de régression utilisés
- LinearRegression
- Ridge
- Lasso
- DecisionTreeRegressor
- RandomForestRegressor
- GradientBoostingRegressor
- XGBRegressor (XGBoost)

### 8.6 Validation
- Cross-validation:
  - StratifiedKFold (classification)
  - KFold (régression)

## 9) Classification, régression et prévision temporelle
### Classification implémentée
Cibles observées dans les scripts:
- client_risque
- installation_reussie_encoded
- client_satisfait_encoded
- client_a_reclame_encoded

### Régression implémentée
- Duree_installation_minutes

### Prévision temporelle
- Aucune implémentation ARIMA/SARIMA détectée dans le code.
- Aucune implémentation Prophet/LSTM détectée.

## 10) Évaluation et comparaison des modèles
Métriques effectivement utilisées:
- Classification: Accuracy, F1-score, ROC-AUC, CV F1
- Régression: R², MAE, RMSE, CV R²

Outils d'analyse des performances:
- Classification report (scikit-learn)
- Matrice de confusion
- Graphiques réel vs prédit et analyse de résidus (régression)

Fine-tuning implémenté dans analyse_copy.py:
- GridSearchCV sur:
  - Gradient Boosting, Random Forest, XGBoost (classification/régression)
  - KNN, SVM (classification)
  - Linear Regression, Ridge, Lasso (régression)

## 11) Interprétabilité des modèles et analyse des variables importantes
Interprétabilité implémentée (analyse_copy.py et SAD.ipynb):
- SHAP avec TreeExplainer sur le modèle de risque.
- SHAP summary plot.
- Classement des variables par impact moyen absolu SHAP.
- Fonctions utilitaires pour formater les causes (shap_json, shap_ui_format).
- Génération de recommandations et priorisation d'actions à partir des causes et de la probabilité de risque.

Sortie orientée dashboard (en mémoire):
- Construction d'une table df_dashboard avec client_id, risque, causes et actions.

## 12) Technologies et bibliothèques utilisées
Bibliothèques Python réellement importées dans le dépôt:
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn
- xgboost
- plotly
- category-encoders
- imbalanced-learn (import présent dans certains blocs)
- shap

## 13) Installation et configuration de l'environnement
### 13.1 Prérequis
- Python 3.10+ recommandé
- Environnement virtuel recommandé

### 13.2 Installation
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost plotly category-encoders imbalanced-learn shap jupyter
```

### 13.3 Encodage et format de données
- Les CSV utilisent le séparateur ; et l'encodage cp1252 dans les scripts.

## 14) Instructions d'exécution des différents modules
### Option A - Exécution notebook (recommandée pour l'exploration)
```bash
jupyter notebook
```
Puis ouvrir:
- analyse_copy.ipynb (pipeline complet + SAD)
- SAD.ipynb (orientation aide à la décision)
- analyse.ipynb (version alternative historique)

### Option B - Exécution script
```bash
python analyse_copy.py
```
Ce script contient:
- nettoyage + imputation + export installations_produits_CNI_clean.csv
- EDA/KPI
- modélisation et interprétabilité

Autres scripts exécutables (variantes):
```bash
python analysecopy.py
python analyse.py
```
Attention:
- Ces variantes peuvent contenir des blocs historiques/commentés et ne représentent pas toutes le même niveau de complétude.
- Le script analyse.py référence également un fichier base_de_données_installations_produits_CNI_2.csv dans une cellule exportée, non présent dans cette arborescence.

## 15) Organisation globale des deux dépôts: AI & Data Science et Web Development
État observé ici:
- Ce dépôt correspond au volet AI & Data Science.
- Le volet Web Development n'est pas présent dans cette arborescence.

Contrat d'intégration suggéré (aligné sur le code existant):
- Entrée Web -> AI: données client/installation.
- Sortie AI -> Web: score de risque, causes principales (SHAP), actions recommandées (structure type df_dashboard).

```mermaid
flowchart LR
    A[Web Development] -->|requêtes données installation| B[AI & Data Science]
    B -->|risque + causes + actions| A
```

## 16) Perspectives d'amélioration
Axes d'amélioration cohérents avec le code actuel:
- Industrialiser le pipeline en modules Python réutilisables (au lieu d'exports notebook volumineux).
- Ajouter une couche de versionnement des jeux de données et de tracking d'expériences.
- Fiabiliser l'exécution bout-en-bout via scripts CLI dédiés (clean, train, explain, export).
- Ajouter des tests unitaires sur les règles d'imputation et transformations.
- Standardiser la persistance des artefacts (modèles, features retenues, rapports).
- Mettre en place une API d'inférence pour la consommation côté Web.
- Ajouter la prévision temporelle uniquement si besoin métier explicite (ARIMA/SARIMA non implémentés actuellement).

---

## Annexe - Jeux de données présents
Tous les fichiers CSV partagent le même schéma de colonnes métier (20 colonnes) avec séparateur ;.

| Fichier | Lignes (approx.) | Rôle observé |
|---|---:|---|
| installations_produits_CNI.csv | 15 273 | Source principale brute dans analyse_copy.py |
| installations_produits_CNI_clean.csv | 14 741 | Sortie nettoyée produite par analyse_copy.py |
| base_finale_v3.csv | 15 151 | Source principale dans analyse.py / analysecopy.py |
| base_finale_v4.csv | 15 273 | Variante de base |
| base_de_donnees_installations_produits_CNI.csv | 15 273 | Variante de base |
| base_de_données_installations_produits_CNI_4.csv | 14 401 | Variante intermédiaire/finale |
