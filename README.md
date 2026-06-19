# Analyse de mes courses Strava

Application interactive : https://samuel-strava-analyse.streamlit.app

## Description

Analyse exploratoire et modélisation prédictive de mes données personnelles de course
à pied extraites via l'API Strava. Ce projet démontre l'extraction via API REST (OAuth 2.0),
le nettoyage ETL, la visualisation interactive, la modélisation Machine Learning et la
génération de plans d'entraînement par RAG (Retrieval-Augmented Generation) — le tout
sur des données personnelles réelles.

## Application Streamlit — 3 pages

### 1. Analyse des courses
- 720+ courses analysées depuis juillet 2023
- Visualisations interactives Plotly (distance, allure, progression annuelle)
- Carte GPS Folium de toutes les courses (596 courses géolocalisées)
- Simulateur d'allure par modèle Random Forest
- Section Insights : découvertes clés tirées des données

### 2. Générateur de plan d'entraînement (RAG)
- Profil automatique basé sur les 20 dernières courses Strava
- RAG : exemples réels tirés de 73 semaines de plans de coach (557 séances)
- Génération via Groq API (Llama 3.3-70b) avec prompt style coach québécois
- Volume cible calculé automatiquement selon l'objectif et les semaines avant course

### 3. Prévision ML vs Performances réelles
- Détection automatique des marathons officiels (distance > 40 km, FC > 155 bpm)
- Comparaison prédiction Random Forest vs temps réels avec vraie FC de chaque course
- Progression sur 3 marathons : 3h38 (2024) → 3h00 (2025) → 2h53 (2026)
- Se met à jour automatiquement à chaque nouveau marathon

## Aperçu des données

- 720+ courses analysées depuis juillet 2023
- 13.0 km de distance moyenne par course
- 9331 km au total
- 5:19 min/km allure moyenne
- 4787 km courus en 2025 uniquement

## Graphiques

### Vue générale
![Analyse overview](strava_overview.png)

### Volume hebdomadaire
![Volume hebdomadaire](volume_hebdo.png)

### Progression annuelle
![Progression annuelle](progression_annuelle.png)

## Modèle Machine Learning

Random Forest Regressor entraîné sur 80% des courses, testé sur 20%.

- **R² : 0.813** (81% de la variation d'allure expliquée)
- **MAE : 0.150 min/km** (erreur moyenne de 9 secondes par km)
- **6 variables** : distance, FC, dénivelé, année, mois, jour de semaine

### Découverte clé
La fréquence cardiaque explique 70% de l'allure — confirmant scientifiquement
l'importance de l'entraînement par zones cardiaques.

### Limite du modèle — l'humain derrière la machine
Le modèle prédit précisément les sorties d'entraînement mais ne capture pas
l'adrénaline de compétition, la préparation spécifique ni les chaussures carbone.
Écart moyen en marathon officiel : moins de 3 minutes — ce qui démontre la
robustesse du modèle quand alimenté avec la vraie FC de course.

## Pipeline de données automatisé

GitHub Actions tourne chaque lundi à 8h UTC et :
1. Rafraîchit le token OAuth Strava
2. Télécharge toutes les activités via l'API REST
3. Nettoie et transforme les données (ETL Pandas)
4. Met à jour `strava_courses.csv` et `strava_gps.csv`
5. Génère les graphiques de synthèse
6. Push automatiquement sur GitHub → Streamlit se redéploie

## Technologies utilisées

- **Python** : Pandas, Plotly, Scikit-learn, Folium, Streamlit
- **API Strava** : Authentification OAuth 2.0, endpoints REST
- **Machine Learning** : Random Forest Regressor (Scikit-learn)
- **RAG** : Groq API (Llama 3.3-70b-versatile), retrieval sur CSV
- **Infrastructure** : GitHub Actions (automatisation hebdomadaire), Streamlit Cloud
- **Versionnement** : Git / GitHub

## Auteur

Samuel Paul — Étudiant AEC Science des données et BI  
GitHub : https://github.com/samuelpauldata  
Application : https://samuel-strava-analyse.streamlit.app