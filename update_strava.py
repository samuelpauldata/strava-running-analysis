import requests
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

# ---- Rafraîchir le token automatiquement ----
def rafraichir_token():
    response = requests.post(
        "https://www.strava.com/oauth/token",
        data={
            "client_id": os.environ["CLIENT_ID"],
            "client_secret": os.environ["CLIENT_SECRET"],
            "refresh_token": os.environ["REFRESH_TOKEN"],
            "grant_type": "refresh_token"
        }
    )
    tokens = response.json()
    return tokens["access_token"]

# ---- Télécharger toutes les courses ----
def telecharger_courses(access_token):
    headers = {"Authorization": f"Bearer {access_token}"}
    activites = []
    page = 1
    while True:
        response = requests.get(
            "https://www.strava.com/api/v3/athlete/activities",
            headers=headers,
            params={"per_page": 100, "page": page}
        )
        data = response.json()
        if not data:
            break
        activites.extend(data)
        page += 1
    return activites

# ---- Nettoyer les données ----
def nettoyer_donnees(activites):
    df = pd.DataFrame(activites)
    df = df[df["type"] == "Run"].copy()
    colonnes = ["name","start_date_local","distance","moving_time",
                "total_elevation_gain","average_speed","average_heartrate",
                "max_heartrate","suffer_score"]
    colonnes_dispo = [c for c in colonnes if c in df.columns]
    df = df[colonnes_dispo]
    df["distance_km"] = df["distance"] / 1000
    df["duree_min"] = df["moving_time"] / 60
    df["allure_min_km"] = df["duree_min"] / df["distance_km"]
    df["start_date_local"] = pd.to_datetime(df["start_date_local"])
    return df

# ---- Générer les graphiques ----
def generer_graphiques(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Analyse de mes courses Strava — Mis à jour le {datetime.now().strftime('%Y-%m-%d')}", 
                 fontsize=14, fontweight="bold")

    axes[0,0].plot(df["start_date_local"], df["distance_km"], alpha=0.5, color="orangered")
    axes[0,0].set_title("Distance par course (km)")
    axes[0,0].set_xlabel("Date")
    axes[0,0].set_ylabel("Distance (km)")

    axes[0,1].hist(df["distance_km"], bins=30, color="orangered", alpha=0.7)
    axes[0,1].set_title("Distribution des distances")
    axes[0,1].set_xlabel("Distance (km)")
    axes[0,1].set_ylabel("Nombre de courses")

    axes[1,0].plot(df["start_date_local"], df["allure_min_km"], alpha=0.5, color="steelblue")
    axes[1,0].set_title("Évolution de l allure (min/km)")
    axes[1,0].set_xlabel("Date")
    axes[1,0].set_ylabel("Allure (min/km)")
    axes[1,0].invert_yaxis()

    axes[1,1].scatter(df["distance_km"], df["average_heartrate"], alpha=0.4, color="green")
    axes[1,1].set_title("Fréquence cardiaque vs Distance")
    axes[1,1].set_xlabel("Distance (km)")
    axes[1,1].set_ylabel("FC moyenne (bpm)")

    plt.tight_layout()
    plt.savefig("strava_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Graphique généré ✅")

# ---- Programme principal ----
print("Démarrage de la mise à jour...")
access_token = rafraichir_token()
print("Token rafraîchi ✅")
activites = telecharger_courses(access_token)
print(f"{len(activites)} activités téléchargées ✅")
df = nettoyer_donnees(activites)
print(f"{len(df)} courses à pied ✅")
df.to_csv("strava_courses.csv", index=False)
print("CSV sauvegardé ✅")
generer_graphiques(df)
print("Mise à jour complète ✅")
