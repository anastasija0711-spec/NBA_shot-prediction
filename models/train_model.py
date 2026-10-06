import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

print("==================================================")
print("🏀 NBA SHOT PREDICTION - MACHINE LEARNING PIPELINE")
print("==================================================")


curr_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(curr_dir) == 'models':
    root_dir = os.path.dirname(curr_dir)
else:
    root_dir = curr_dir

shots_path = os.path.join(root_dir, 'data', 'shot_logs.csv')
players_path = os.path.join(root_dir, 'data', 'players.csv')
models_dir = os.path.join(root_dir, 'models')

print(f"\n[1/5] Učitavanje baza podataka:")
print(f" - Šutevi: {shots_path}")
print(f" - Igrači: {players_path}")

if not os.path.exists(shots_path) or not os.path.exists(players_path):
    print("Greška: Nisu pronađeni CSV fajlovi u folderu 'data/'.")
    exit(1)

shots_df = pd.read_csv(shots_path)
players_df = pd.read_csv(players_path)


print("\n[2/5] Priprema podataka i Feature Engineering...")

shots_df['player_name'] = shots_df['player_name'].astype(str).str.lower().str.strip()
players_df['Name'] = players_df['Name'].astype(str).str.lower().str.strip()

# Uklanjanje duplikata igraca
players_clean = players_df.drop_duplicates(subset=['Name'], keep='last')
zeljene_kolone = [c for c in ['Name', 'Pos', 'Height', 'Experience'] if c in players_clean.columns]
players_clean = players_clean[zeljene_kolone]

# Spajanje
df = pd.merge(shots_df, players_clean, left_on='player_name', right_on='Name', how='left')

def height_to_inches(h_str):
    try:
        if pd.isna(h_str):
            return 78.0
        parts = str(h_str).split('-')
        if len(parts) == 2:
            return int(parts[0]) * 12 + int(parts[1])
        return 78.0
    except:
        return 78.0

df['Height_Inches'] = df['Height'].apply(height_to_inches)
df['Experience_Num'] = pd.to_numeric(df['Experience'], errors='coerce').fillna(4.0)
df['Is_Home'] = (df['LOCATION'].astype(str).str.upper() == 'H').astype(int)

median_shot_clock = df['SHOT_CLOCK'].median() if 'SHOT_CLOCK' in df.columns else 12.0
df['SHOT_CLOCK'] = df['SHOT_CLOCK'].fillna(median_shot_clock)

median_def_dist = df['CLOSE_DEF_DIST'].median() if 'CLOSE_DEF_DIST' in df.columns else 4.0
df['CLOSE_DEF_DIST'] = df['CLOSE_DEF_DIST'].fillna(median_def_dist)

df['DRIBBLES'] = pd.to_numeric(df['DRIBBLES'], errors='coerce').fillna(0)
df['TOUCH_TIME'] = pd.to_numeric(df['TOUCH_TIME'], errors='coerce').fillna(1.5)

# CILJNA PROMENLJIVA (TARGET): FGM (1 = pogodak, 0 = promašaj)
df['FGM'] = pd.to_numeric(df['FGM'], errors='coerce').fillna(0).astype(int)

features = [
    'SHOT_DIST',
    'CLOSE_DEF_DIST',
    'SHOT_CLOCK',
    'DRIBBLES',
    'TOUCH_TIME',
    'PTS_TYPE',
    'Is_Home',
    'Height_Inches',
    'Experience_Num'
]

ml_data = df.dropna(subset=features + ['FGM'])
X = ml_data[features]
y = ml_data['FGM']

print(f"Dostupno uzoraka za treniranje: {len(X)} šuteva.")
print(f"Pogoci: {y.sum()} ({y.mean()*100:.1f}%), Promašaji: {len(y) - y.sum()}")

# treniranje modela
print("\n[3/5] Deljenje na Train (80%) i Test (20%) skupove...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("[4/5] Treniranje Random Forest Classifier modela...")
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    min_samples_split=20,
    random_state=42, 
    n_jobs=-1
)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*50)
print("📊 REZULTATI I PERFORMANSE MODELA")
print("="*50)
print(f"Tačnost modela (Accuracy): {acc * 100:.2f}%")
print(f"ROC-AUC skor:              {roc_auc:.4f}")

print("\nMatrica konfuzije:")
print(confusion_matrix(y_test, y_pred))

print("\nKlasifikacioni izveštaj:")
print(classification_report(y_test, y_pred, target_names=['Promašaj (0)', 'Pogodak (1)']))

print("Ključni faktori koji najviše utiču na pogodak (Feature Importance):")
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
for feat, imp in importances.items():
    print(f"  - {feat:16s}: {imp * 100:.2f}%")


os.makedirs(models_dir, exist_ok=True)
model_file = os.path.join(models_dir, 'nba_shot_model.joblib')
joblib.dump(model, model_file)
print(f"\n[5/5] Model je uspešno snimljen u: '{model_file}'")


print("\n" + "="*50)
print(" TESTIRANJE MODELA UŽIVO (PREDIKCIJA VEROVATNOĆE)")
print("="*50)

def simuliraj_sut():
    print("\nUnesi parametre situacije na terenu:")
    try:
        shot_dist = float(input("1. Distanca šuta u stopama (npr. 4 za obruč, 24 za trojku): "))
        pts_type = 3 if shot_dist >= 22.5 else 2
        close_def = float(input("2. Blizina defanzivca u stopama (npr. 1.5 tesno, 5.0 otvoren): "))
        shot_clock = float(input("3. Vreme na satu napada u sekundama (1 do 24): "))
        dribbles = int(input("4. Broj driblinga pre šuta (0, 1, 5...): "))
        touch_time = float(input("5. Vreme kontakta s loptom u sekundama (npr. 1.2): "))
        is_home = int(input("6. Domaći teren? (1 = Domaći, 0 = Gostujući): "))
        height_in = float(input("7. Visina igrača u inčima (npr. 80 za 6-8 kao LeBron): "))
        exp = float(input("8. Godine iskustva u ligi (npr. 5): "))

        input_data = pd.DataFrame([{
            'SHOT_DIST': shot_dist,
            'CLOSE_DEF_DIST': close_def,
            'SHOT_CLOCK': shot_clock,
            'DRIBBLES': dribbles,
            'TOUCH_TIME': touch_time,
            'PTS_TYPE': pts_type,
            'Is_Home': is_home,
            'Height_Inches': height_in,
            'Experience_Num': exp
        }])

        proba = model.predict_proba(input_data)[0][1]
        predikcija = "POGODAK" if proba >= 0.5 else "PROMAŠAJ"

        print("\n--- REZULTAT MODELA ---")
        print(f"Predikcija modela: {predikcija}")
        print(f"Verovatnoća da ulazi koš: {proba * 100:.2f}%")
        expected_points = proba * pts_type
        print(f"Očekivani poeni po ovom šutu (xP): {expected_points:.2f} poena")
        print("-----------------------")
    except Exception as e:
        print(f"Greška pri unosu: {e}")

while True:
    simuliraj_sut()
    jos = input("\nŽeliš li da testiraš još jednu situaciju na terenu? (da/ne): ").strip().lower()
    if jos != 'da':
        print("\nKraj programa. Tvoj ML model je spreman!")
        break