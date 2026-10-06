import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    roc_auc_score, 
    confusion_matrix, 
    log_loss
)

def main():
    print("=" * 60)
    print("NBA SHOT PREDICTION")
    print("=" * 60)

    # definisanje putanja do podataka
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(curr_dir) if os.path.basename(curr_dir) == 'models' else curr_dir
    shots_path = os.path.join(root_dir, 'data', 'shot_logs.csv')
    players_path = os.path.join(root_dir, 'data', 'players.csv')
    models_dir = os.path.join(root_dir, 'models')

    print(f"\n[1/5] Učitavanje baza podataka:")
    print(f" - Šutevi: {shots_path}")
    print(f" - Igrači: {players_path}")

    if not os.path.exists(shots_path) or not os.path.exists(players_path):
        print("Greška: Nisu pronađeni CSV fajlovi u folderu 'data/'.")
        return

    # Ucitavanje podataka
    shots_df = pd.read_csv(shots_path)
    players_df = pd.read_csv(players_path)

    print("\n[2/5] Priprema podataka i Feature Engineering...")

    def detect_id_column(df, candidates=['player_id', 'PLAYER_ID', 'PlayerID', 'Person_id', 'id']):
        for col in candidates:
            if col in df.columns:
                return col
        for col in df.columns:
            if 'id' in col.lower() and 'game' not in col.lower() and 'team' not in col.lower():
                return col
        return None

    shot_id_col = detect_id_column(shots_df)
    player_id_col = detect_id_column(players_df)

    if not shot_id_col or not player_id_col:
        print(" ID kolona nije pronađena eksplicitno, kreira se ID na osnovu imena...")
        shots_df['player_id'] = shots_df['player_name'].astype(str).str.lower().str.strip()
        players_df['player_id'] = players_df['Name'].astype(str).str.lower().str.strip()
        shot_id_col = 'player_id'
        player_id_col = 'player_id'
    else:
        shots_df[shot_id_col] = pd.to_numeric(shots_df[shot_id_col], errors='coerce').fillna(-1).astype(int)
        players_df[player_id_col] = pd.to_numeric(players_df[player_id_col], errors='coerce').fillna(-1).astype(int)

    
    shots_df['FGM'] = pd.to_numeric(shots_df['FGM'], errors='coerce').fillna(0).astype(int)
    
    player_career_stats = shots_df.groupby(shot_id_col).agg(
        career_total_shots=('FGM', 'count'),
        career_made_shots=('FGM', 'sum')
    ).reset_index()

    LEAGUE_AVG = 0.45
    WEIGHT = 25
    player_career_stats['career_fg_pct'] = player_career_stats['career_made_shots'] / player_career_stats['career_total_shots'].replace(0, 1)
    player_career_stats['career_fg_smoothed'] = (
        (player_career_stats['career_made_shots'] + WEIGHT * LEAGUE_AVG) / 
        (player_career_stats['career_total_shots'] + WEIGHT)
    )

    team_col = next((c for c in ['TEAM_ABBR', 'team', 'Team', 'Tm', 'TEAM_ID'] if c in shots_df.columns), None)
    if team_col:
        clubs_by_player = shots_df.groupby(shot_id_col)[team_col].unique().reset_index()
        clubs_by_player['all_clubs'] = clubs_by_player[team_col].apply(lambda x: [str(i) for i in x])
        player_career_stats = pd.merge(player_career_stats, clubs_by_player[[shot_id_col, 'all_clubs']], on=shot_id_col, how='left')
    else:
        player_career_stats['all_clubs'] = [[] for _ in range(len(player_career_stats))]

    
    def parse_height(h):
        if pd.isna(h): return 78.0
        try:
            ft, inc = str(h).split('-')
            return int(ft) * 12 + int(inc)
        except:
            return 78.0

    players_df['Height_Inches'] = players_df['Height'].apply(parse_height) if 'Height' in players_df.columns else 78.0
    players_df['Experience_Num'] = pd.to_numeric(players_df['Experience'], errors='coerce').fillna(4.0) if 'Experience' in players_df.columns else 4.0
    
    name_col = 'Name' if 'Name' in players_df.columns else ('player_name' if 'player_name' in players_df.columns else player_id_col)
    players_df[name_col] = players_df[name_col].astype(str).str.lower().str.strip()

    player_profiles = players_df.groupby(player_id_col).agg({
        name_col: 'last',
        'Height_Inches': 'last',
        'Experience_Num': 'max'
    }).reset_index().rename(columns={name_col: 'player_name_clean'})

    # spajanje tabela
    df = pd.merge(shots_df, player_profiles, left_on=shot_id_col, right_on=player_id_col, how='left')
    df = pd.merge(df, player_career_stats[[shot_id_col, 'career_fg_smoothed', 'career_fg_pct', 'career_total_shots']], on=shot_id_col, how='left')

    df['Height_Inches'] = df['Height_Inches'].fillna(78.0)
    df['Experience_Num'] = df['Experience_Num'].fillna(4.0)
    df['Is_Home'] = (df['LOCATION'].astype(str).str.upper() == 'H').astype(int) if 'LOCATION' in df.columns else 1

    df['SHOT_CLOCK'] = df['SHOT_CLOCK'].fillna(df['SHOT_CLOCK'].median() if 'SHOT_CLOCK' in df.columns else 12.0)
    df['CLOSE_DEF_DIST'] = df['CLOSE_DEF_DIST'].fillna(df['CLOSE_DEF_DIST'].median() if 'CLOSE_DEF_DIST' in df.columns else 4.0)
    df['DRIBBLES'] = pd.to_numeric(df['DRIBBLES'], errors='coerce').fillna(0)
    df['TOUCH_TIME'] = pd.to_numeric(df['TOUCH_TIME'], errors='coerce').fillna(1.5)
    df['career_fg_smoothed'] = df['career_fg_smoothed'].fillna(LEAGUE_AVG)

    features = [
        'SHOT_DIST', 
        'CLOSE_DEF_DIST', 
        'SHOT_CLOCK', 
        'DRIBBLES', 
        'TOUCH_TIME', 
        'PTS_TYPE', 
        'Is_Home', 
        'Height_Inches', 
        'Experience_Num',
        'career_fg_smoothed'
    ]

    ml_data = df.dropna(subset=features + ['FGM'])
    X = ml_data[features]
    y = ml_data['FGM']

    print(f"Dostupno uzoraka za treniranje: {len(X)} šuteva.")
    print(f"Pogoci: {y.sum()} ({y.mean()*100:.1f}%), Promašaji: {len(y) - y.sum()} ({(1-y.mean())*100:.1f}%)")

    
    print("\n[3/5] Deljenje na Train (80%) i Test (20%) uz stratifikaciju...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # optimizacija hiperparametara pomoću GridSearchCV
    print("\n[4/5] Pokretanje GridSearchCV optimizacije hiperparametara (3-fold CV)...")
    print("Mreža hiperparametara za pretragu:")
    print(" - n_estimators: [50, 100, 150]")
    print(" - max_depth: [6, 10, 14]")
    print(" - min_samples_split: [10, 20, 30]")
    print("Napomena: Proces može potrajati par minuta dok isproba sve kombinacije...")

    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [6, 10, 14],
        'min_samples_split': [10, 20, 30]
    }

    base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=3,
        scoring='roc_auc',  
        n_jobs=-1,
        verbose=1
    )

    # Kros-validacija
    grid_search.fit(X_train, y_train)

    print("\n" + "=" * 55)
    print("🏆 NAJBOLJI REZULTATI GRID SEARCH-A")
    print("=" * 55)
    print(f"Najbolji parametri (best_params_):     {grid_search.best_params_}")
    print(f"Najbolji CV ROC-AUC skor:              {grid_search.best_score_:.4f}")

    # Pobednicki model
    model = grid_search.best_estimator_

    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    test_accuracy = accuracy_score(y_test, y_pred)
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    test_log_loss = log_loss(y_test, y_pred_proba)

    print("\n" + "=" * 55)
    print(" PERFORMANSE NA NEVIĐENOM TEST SKUPU")
    print("=" * 55)
    print(f"Tačnost modela (Accuracy):              {test_accuracy * 100:.2f}%")
    print(f"ROC-AUC skor:                           {test_roc_auc:.4f}")
    print(f"Log-Loss (Binary Cross-Entropy):        {test_log_loss:.4f}")
    print("=" * 55)

    print("\nMatrica konfuzije:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  [TN: {cm[0][0]:5d} | FP: {cm[0][1]:5d}]")
    print(f"  [FN: {cm[1][0]:5d} | TP: {cm[1][1]:5d}]")

    print("\nKlasifikacioni izveštaj:")
    print(classification_report(y_test, y_pred, target_names=['Promašaj (0)', 'Pogodak (1)']))

    print("\nKljučni faktori koji najviše utiču na pogodak (Feature Importance):")
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    for feat, imp in importances.items():
        print(f"  - {feat:20s}: {imp * 100:.2f}%")

    
    os.makedirs(models_dir, exist_ok=True)
    model_file = os.path.join(models_dir, 'nba_shot_model.joblib')
    joblib.dump(model, model_file)
    
    full_player_info = pd.merge(player_profiles, player_career_stats, left_on=player_id_col, right_on=shot_id_col, how='inner')
    player_id_map = {}
    for _, row in full_player_info.iterrows():
        pid = row[player_id_col]
        player_id_map[str(pid).lower()] = {
            'name': row.get('player_name_clean', 'Nepoznat'),
            'height': row.get('Height_Inches', 78.0),
            'exp': row.get('Experience_Num', 4.0),
            'career_fg_smoothed': row.get('career_fg_smoothed', LEAGUE_AVG),
            'career_fg_pct': row.get('career_fg_pct', LEAGUE_AVG),
            'clubs': row.get('all_clubs', [])
        }
    joblib.dump(player_id_map, os.path.join(models_dir, 'player_id_map.joblib'))
    
    print(f"\n[5/5] Optimizovani model je uspešno snimljen u: '{model_file}'")

    # Testiranje
    print("\n" + "=" * 55)
    print(" TESTIRANJE OPTIMIZOVANOG MODELA UŽIVO")
    print("=" * 55)

    while True:
        print("\n" + "-" * 40)
        target_id = input("Unesite ID igrača ili Ime (ili 'q' za izlaz): ").strip().lower()
        if target_id in ['q', 'exit', 'izlaz']:
            print("\nKraj rada. Tvoj optimizovani ML model je spreman!")
            break

        if target_id in player_id_map:
            p = player_id_map[target_id]
            print(f"   Pronađen igrač: {p['name'].title()}")
            print(f"   Klubovi: {p['clubs'] if p['clubs'] else 'Evidentirani u bazi'}")
            print(f"   Karijerni % šuta: {p['career_fg_pct']*100:.1f}%")
            
            height_in = p['height']
            exp = p['exp']
            career_fg = p['career_fg_smoothed']
        else:
            print(f"  Igrač '{target_id}' nije u bazi. Koristiće se prosek lige (6'6\", 4 god iskustva).")
            height_in = 78.0
            exp = 4.0
            career_fg = LEAGUE_AVG

        try:
            print("\nUnesite parametre situacije na terenu:")
            shot_dist = float(input("1. Distanca šuta u stopama (npr. 4 za obruč, 24 za trojku): "))
            pts_type = 3 if shot_dist >= 22.5 else 2
            close_def = float(input("2. Blizina defanzivca u stopama (npr. 1.5 tesno, 5.0 otvoren): "))
            shot_clock = float(input("3. Vreme na satu napada u sekundama (1 do 24): "))
            dribbles = int(input("4. Broj driblinga pre šuta: "))
            touch_time = float(input("5. Vreme kontakta s loptom u sekundama (npr. 1.2): "))
            is_home = int(input("6. Domaći teren? (1 = Domaći, 0 = Gostujući): "))

            input_data = pd.DataFrame([{
                'SHOT_DIST': shot_dist,
                'CLOSE_DEF_DIST': close_def,
                'SHOT_CLOCK': shot_clock,
                'DRIBBLES': dribbles,
                'TOUCH_TIME': touch_time,
                'PTS_TYPE': pts_type,
                'Is_Home': is_home,
                'Height_Inches': height_in,
                'Experience_Num': exp,
                'career_fg_smoothed': career_fg
            }])

            proba = model.predict_proba(input_data)[0][1]
            predikcija = "POGODAK" if proba >= 0.5 else "PROMAŠAJ"

            print("\n--- REZULTAT MODELA ---")
            print(f"Predikcija modela:        {predikcija}")
            print(f"Verovatnoća pogotka:      {proba * 100:.2f}%")
            print(f"Očekivani poeni (xP):     {proba * pts_type:.2f} poena")
            print("-----------------------")
        except Exception as e:
            print(f"Greška pri unosu: {e}")

if __name__ == "__main__":
    main()