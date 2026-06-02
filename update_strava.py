import requests
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

def rafraichir_token():
    response = requests.post(
        'https://www.strava.com/oauth/token',
        data={
            'client_id': os.environ['CLIENT_ID'],
            'client_secret': os.environ['CLIENT_SECRET'],
            'refresh_token': os.environ['REFRESH_TOKEN'],
            'grant_type': 'refresh_token'
        }
    )
    return response.json()['access_token']

def telecharger_courses(access_token):
    headers = {'Authorization': f'Bearer {access_token}'}
    activites = []
    page = 1
    while True:
        response = requests.get(
            'https://www.strava.com/api/v3/athlete/activities',
            headers=headers,
            params={'per_page': 100, 'page': page}
        )
        data = response.json()
        if not data:
            break
        activites.extend(data)
        page += 1
    return activites

def nettoyer_donnees(activites):
    df = pd.DataFrame(activites)
    df = df[df['type'] == 'Run'].copy()
    colonnes = ['name','start_date_local','distance','moving_time',
                'total_elevation_gain','average_speed','average_heartrate']
    colonnes_dispo = [c for c in colonnes if c in df.columns]
    df = df[colonnes_dispo]
    df['distance_km'] = df['distance'] / 1000
    df['duree_min'] = df['moving_time'] / 60
    df['allure_min_km'] = df['duree_min'] / df['distance_km']
    df['start_date_local'] = pd.to_datetime(df['start_date_local'], utc=True)
    df['annee'] = df['start_date_local'].dt.year
    return df

def generer_graphiques(df):
    date_maj = datetime.now().strftime('%Y-%m-%d')

    # 1. Vue generale
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Analyse Strava - Mis a jour le {date_maj}', fontsize=14, fontweight='bold')
    axes[0,0].plot(df['start_date_local'], df['distance_km'], alpha=0.5, color='orangered')
    axes[0,0].set_title('Distance par course (km)')
    axes[0,1].hist(df['distance_km'], bins=30, color='orangered', alpha=0.7)
    axes[0,1].set_title('Distribution des distances')
    axes[1,0].plot(df['start_date_local'], df['allure_min_km'], alpha=0.5, color='steelblue')
    axes[1,0].set_title('Allure (min/km)')
    axes[1,0].invert_yaxis()
    axes[1,1].scatter(df['distance_km'], df['average_heartrate'], alpha=0.4, color='green')
    axes[1,1].set_title('FC vs Distance')
    plt.tight_layout()
    plt.savefig('strava_overview.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Volume hebdomadaire
    df['annee_semaine'] = df['start_date_local'].dt.to_period('W')
    volume_hebdo = df.groupby('annee_semaine').agg(
        km_total=('distance_km', 'sum'),
        nb_courses=('distance_km', 'count')
    ).reset_index()
    volume_hebdo['annee_semaine_dt'] = volume_hebdo['annee_semaine'].dt.to_timestamp()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Volume hebdomadaire', fontsize=14, fontweight='bold')
    axes[0].bar(volume_hebdo['annee_semaine_dt'], volume_hebdo['km_total'], color='orangered', alpha=0.7, width=5)
    axes[0].axhline(y=volume_hebdo['km_total'].mean(), color='black', linestyle='--')
    axes[0].set_title('Km par semaine')
    axes[1].bar(volume_hebdo['annee_semaine_dt'], volume_hebdo['nb_courses'], color='steelblue', alpha=0.7, width=5)
    axes[1].set_title('Nombre de courses par semaine')
    plt.tight_layout()
    plt.savefig('volume_hebdo.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Progression annuelle
    stats_annee = df.groupby('annee').agg(
        km_total=('distance_km', 'sum'),
        km_moyen=('distance_km', 'mean'),
        allure_moyenne=('allure_min_km', 'mean')
    ).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Progression annuelle', fontsize=14, fontweight='bold')
    axes[0].bar(stats_annee['annee'], stats_annee['km_total'], color='orangered', alpha=0.8)
    axes[0].set_title('Km total par annee')
    axes[1].bar(stats_annee['annee'], stats_annee['km_moyen'], color='steelblue', alpha=0.8)
    axes[1].set_title('Distance moyenne par course')
    axes[2].bar(stats_annee['annee'], stats_annee['allure_moyenne'], color='green', alpha=0.8)
    axes[2].set_title('Allure moyenne (min/km)')
    axes[2].invert_yaxis()
    plt.tight_layout()
    plt.savefig('progression_annuelle.png', dpi=150, bbox_inches='tight')
    plt.close()

    print('Graphiques generes')

print('Demarrage...')
access_token = rafraichir_token()
print('Token rafraichi')
activites = telecharger_courses(access_token)
print(f'{len(activites)} activites telechargees')
df = nettoyer_donnees(activites)
df.to_csv('strava_courses.csv', index=False)
generer_graphiques(df)
telecharger_gps(access_token)
print('Mise a jour complete!')

def telecharger_gps(access_token):
    headers = {"Authorization": f"Bearer {access_token}"}
    tous_ids = []
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
        for a in data:
            if a["type"] == "Run":
                tous_ids.append({"id": a["id"], "name": a["name"], "start_date_local": a["start_date_local"]})
        page += 1

    traces_gps = []
    import time
    for course in tous_ids:
        response = requests.get(
            f"https://www.strava.com/api/v3/activities/{course['id']}/streams",
            headers=headers,
            params={"keys": "latlng", "key_by_type": "true"}
        )
        data = response.json()
        if "latlng" in data:
            for coord in data["latlng"]["data"]:
                traces_gps.append({
                    "id": course["id"],
                    "nom": course["name"],
                    "date": course["start_date_local"],
                    "lat": coord[0],
                    "lng": coord[1]
                })
        time.sleep(0.3)

    df_gps = pd.DataFrame(traces_gps)
    df_gps = df_gps.groupby("id").apply(lambda x: x.iloc[::10]).reset_index(drop=True)
    df_gps.to_csv("strava_gps.csv", index=False)
    print(f"GPS mis a jour : {len(df_gps)} points")
