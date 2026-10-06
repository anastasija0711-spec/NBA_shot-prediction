import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

def main():
    # 1. Definisanje putanja do podataka
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(curr_dir) if os.path.basename(curr_dir) == 'models' else curr_dir
    shots_path = os.path.join(root_dir, 'data', 'shot_logs.csv')
    players_path = os.path.join(root_dir, 'data', 'players.csv')
    models_dir = os.path.join(root_dir, 'models')

    print(" Pokretanje ML pipeline-a sa pretragom po ID-ju i karijernom statistikom...")
    if not os.path.exists(shots_path) or not os.path.exists(players_path):
        raise FileNotFoundError("Sistemska greška: Nedostaju CSV fajlovi u 'data/' direktorijumu.")

    # 2. Učitavanje podataka
    print(" Učitavanje datasetova...")
    shots_df = pd.read_csv(shots_path)
    players_df = pd.read_csv(players_path)

    # -------------------------------------------------------------------------
    # 2.1. Dinamička detekcija i mapiranje ID kolone u oba skupa podataka
    # -------------------------------------------------------------------------
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

    # Ukoliko u CSV-u ne postoji eksplicitna kolona ID, kreiramo je pouzdano
    if not shot_id_col or not player_id_col:
        print("⚠️ ID kolona nije pronađena eksplicitno, kreira se ID na osnovu imena...")
        shots_df['player_id'] = shots_df['player_name'].astype(str).str.lower().str.strip()
        players_df['player_id'] = players_df['Name'].astype(str).str.lower().str.strip()
        shot_id_col = 'player_id'
        player_id_col = 'player_id'
    else:
        shots_df[shot_id_col] = pd.to_numeric(shots_df[shot_id_col], errors='coerce').fillna(-1).astype(int)
        players_df[player_id_col] = pd.to_numeric(players_df[player_id_col], errors='coerce').fillna(-1).astype(int)

    # -------------------------------------------------------------------------
    # 2.2. Agregacija CELOKUPNE statistike igrača iz SVIH KLUBOVA gde je igrao
    # -------------------------------------------------------------------------
    shots_df['FGM'] = pd.to_numeric(shots_df['FGM'], errors='coerce').fillna(0).astype(int)
    
    # Detekcija naziva tima/kluba ako postoji u šutevima
    team_col = next((c for c in ['TEAM_ABBR', 'team', 'Team', 'Tm', 'TEAM_ID'] if c in shots_df.columns), None)

    # Računamo ukupnu karijernu uspešnost igrača kroz sve klubove i sezone
    player_career_stats = shots_df.groupby(shot_id_col).agg(
        career_total_shots=('FGM', 'count'),
        career_made_shots=('FGM', 'sum'),
        career_avg_dist=('SHOT_DIST', 'mean')
    ).reset_index()

    # Bayesian smoothing: sprečava ekstreme (100% ili 0%) kod igrača sa malo šuteva
    LEAGUE_AVG = 0.45
    WEIGHT = 25
    player_career_stats['career_fg_pct'] = player_career_stats['career_made_shots'] / player_career_stats['career_total_shots'].replace(0, 1)
    player_career_stats['career_fg_smoothed'] = (
        (player_career_stats['career_made_shots'] + WEIGHT * LEAGUE_AVG) / 
        (player_career_stats['career_total_shots'] + WEIGHT)
    )

    # Beleženje svih klubova u kojima je igrač bio
    if team_col:
        clubs_by_player = shots_df.groupby(shot_id_col)[team_col].unique().reset_index()
        clubs_by_player['all_clubs'] = clubs_by_player[team_col].apply(lambda x: [str(i) for i in x])
        player_career_stats = pd.merge(player_career_stats, clubs_by_player[[shot_id_col, 'all_clubs']], on=shot_id_col, how='left')
    else:
        player_career_stats['all_clubs'] = [[] for _ in range(len(player_career_stats))]

    # -------------------------------------------------------------------------
    # 2.3. Agregacija fizičkih podataka igrača (bez brisanja istorije klubova)
    # -------------------------------------------------------------------------
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

    # Umesto drop_duplicates(keep='last') koji je brisao klubove, grupišemo po ID-ju:
    player_profiles = players_df.groupby(player_id_col).agg({
        name_col: 'last',
        'Height_Inches': 'last',
        'Experience_Num': 'max'
    }).reset_index().rename(columns={name_col: 'player_name_clean'})

    # -------------------------------------------------------------------------
    # 2.4. Spajanje podataka preko ID-ja
    # -------------------------------------------------------------------------
    # Spajamo šuteve sa profilom i sa celokupnom karijernom statistikom iz svih klubova
    df = pd.merge(shots_df, player_profiles, left_on=shot_id_col, right_on=player_id_col, how='left')
    df = pd.merge(df, player_career_stats[[shot_id_col, 'career_fg_smoothed', 'career_fg_pct', 'career_total_shots']], on=shot_id_col, how='left')

    # 3. Feature Engineering
    df['Height_Inches'] = df['Height_Inches'].fillna(78.0)
    df['Experience_Num'] = df['Experience_Num'].fillna(4.0)
    df['Is_Home'] = (df['LOCATION'].astype(str).str.upper() == 'H').astype(int) if 'LOCATION' in df.columns else 1

    df['SHOT_CLOCK'] = df['SHOT_CLOCK'].fillna(df['SHOT_CLOCK'].median())
    df['CLOSE_DEF_DIST'] = df['CLOSE_DEF_DIST'].fillna(df['CLOSE_DEF_DIST'].median())
    df['DRIBBLES'] = pd.to_numeric(df['DRIBBLES'], errors='coerce').fillna(0)
    df['TOUCH_TIME'] = pd.to_numeric(df['TOUCH_TIME'], errors='coerce').fillna(1.5)

    # Novo ključno obeležje: Individualna uspešnost šutera kroz celu karijeru
    df['career_fg_smoothed'] = df['career_fg_smoothed'].fillna(LEAGUE_AVG)

    # 4. Priprema za treniranje
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
        'career_fg_smoothed'  # Obuhvata sve prethodne klubove igrača
    ]

    df = df.dropna(subset=features + ['FGM'])
    X = df[features]
    y = df['FGM']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 5. Modelovanje
    print(f" Treniranje Random Forest modela na {len(X_train)} uzoraka...")
    model = RandomForestClassifier(n_estimators=120, max_depth=12, min_samples_split=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 6. Evaluacija
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\n" + "-"*40)
    print(" IZVEŠTAJ O PERFORMANSAMA MODELA")
    print("-"*40)
    print(f" Tačnost (Accuracy):  {accuracy_score(y_test, preds):.4f}")
    print(f" ROC-AUC skor:        {roc_auc_score(y_test, probs):.4f}")
    print("\n Klasifikacioni izveštaj:")
    print(classification_report(y_test, preds, target_names=['Promašaj', 'Pogodak']))

    # 7. Čuvanje modela i baze igrača po ID-ju
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, 'nba_model.joblib'))

    # Rečnik za brzu pretragu po ID-ju tokom interaktivnog rada
    full_player_info = pd.merge(player_profiles, player_career_stats, left_on=player_id_col, right_on=shot_id_col, how='inner')
    player_id_map = {}
    for _, row in full_player_info.iterrows():
        pid = row[player_id_col]
        player_id_map[pid] = {
            'name': row.get('player_name_clean', 'Nepoznat'),
            'height': row.get('Height_Inches', 78.0),
            'exp': row.get('Experience_Num', 4.0),
            'career_fg_smoothed': row.get('career_fg_smoothed', LEAGUE_AVG),
            'career_fg_pct': row.get('career_fg_pct', LEAGUE_AVG),
            'total_shots': row.get('career_total_shots', 0),
            'clubs': row.get('all_clubs', [])
        }

    joblib.dump(player_id_map, os.path.join(models_dir, 'player_id_map.joblib'))

    # 8. Interaktivno testiranje pretragom po ID-ju
    print("\n" + "="*45)
    print(" PRETRAGA PO ID-JU IGRAČA I TESTIRANJE")
    print("="*45)

    while True:
        try:
            raw_input_id = input("\nUnesite ID igrača (ili 'q' za izlaz): ").strip()
            if raw_input_id.lower() in ['q', 'exit', 'izlaz']:
                break

            # Podrška za numerički ili string ID
            try:
                target_id = int(raw_input_id)
            except ValueError:
                target_id = raw_input_id

            if target_id in player_id_map:
                p = player_id_map[target_id]
                print(f" Pronađen igrač: {p['name']} (ID: {target_id})")
                print(f" Svi klubovi u kojima je igrao: {p['clubs'] if p['clubs'] else 'Evidentirani u bazi'}")
                print(f" Ukupno šuteva u karijeri: {p['total_shots']}")
                print(f" Ukupan procenat uspešnosti (svi klubovi): {p['career_fg_pct']*100:.1f}%")
                
                height_in = p['height']
                exp = p['exp']
                career_fg = p['career_fg_smoothed']
            else:
                print(f"⚠️ ID {target_id} nije pronađen. Koriste se prosečne ligaške vrednosti.")
                height_in = 78.0
                exp = 4.0
                career_fg = LEAGUE_AVG

            # Parametri konkretnog šuta
            print("\nUnesite parametre šuta:")
            shot_dist = float(input(" - Distanca šuta (ft): "))
            pts_type = 3 if shot_dist >= 22.5 else 2
            close_def = float(input(" - Udaljenost odbrane (ft): "))
            shot_clock = float(input(" - Preostalo vreme napada (s): "))
            dribbles = int(input(" - Broj driblinga: "))
            touch_time = float(input(" - Vreme kontakta sa loptom (s): "))
            is_home = int(input(" - Domaći teren (1=Da, 0=Ne): "))

            inp = pd.DataFrame([[
                shot_dist, close_def, shot_clock, dribbles, touch_time,
                pts_type, is_home, height_in, exp, career_fg
            ]], columns=features)

            p_make = model.predict_proba(inp)[0][1]
            print("-" * 35)
            print(f" -> Predviđena verovatnoća pogotka: {p_make*100:.1f}%")
            print(f" -> Očekivani poeni (xP): {p_make*pts_type:.2f}")
            print("-" * 35)

        except KeyboardInterrupt:
            print("\nZavršetak.")
            break
        except Exception as e:
            print(f"Greška ({e}). Pokušajte ponovo.")

if __name__ == "__main__":
    main()