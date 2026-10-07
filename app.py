# -*- coding: utf-8 -*-


import os
import sys
import json
import webbrowser
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Подешавање фолдера и путања
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir) if os.path.basename(curr_dir) == 'models' else curr_dir

shots_path = os.path.join(root_dir, 'data', 'shot_logs.csv')
players_path = os.path.join(root_dir, 'data', 'players.csv')
models_dir = os.path.join(root_dir, 'models')
model_path = os.path.join(models_dir, 'nba_shot_model.joblib')

if not os.path.exists(shots_path) or not os.path.exists(players_path):
    print("Грешка: Фајлови 'shot_logs.csv' и 'players.csv' морају бити у фолдеру 'data/'.")
    sys.exit(1)

print(" [1/4] Учитавање базе података...")
shots_df = pd.read_csv(shots_path)
players_df = pd.read_csv(players_path)

shots_df['player_name'] = shots_df['player_name'].astype(str).str.lower().str.strip()
name_col = 'Name' if 'Name' in players_df.columns else ('player_name' if 'player_name' in players_df.columns else players_df.columns[0])
players_df[name_col] = players_df[name_col].astype(str).str.lower().str.strip()
players_clean = players_df.drop_duplicates(subset=[name_col], keep='last')

desired_cols = [k for k in [name_col, 'Pos', 'Height', 'Experience'] if k in players_clean.columns]
players_clean = players_clean[desired_cols]
df = pd.merge(shots_df, players_clean, left_on='player_name', right_on=name_col, how='left')

def height_to_inches(h_str):
    try:
        parts = str(h_str).split('-')
        if len(parts) == 2:
            return float(int(parts[0]) * 12 + int(parts[1]))
        return 78.0
    except:
        return 78.0

df['Height_Inches'] = df['Height'].apply(height_to_inches) if 'Height' in df.columns else 78.0
df['Experience_Num'] = pd.to_numeric(df['Experience'], errors='coerce').fillna(4.0) if 'Experience' in df.columns else 4.0
df['Is_Home'] = (df['LOCATION'].astype(str).str.upper() == 'H').astype(int) if 'LOCATION' in df.columns else 1
df['SHOT_CLOCK'] = pd.to_numeric(df['SHOT_CLOCK'], errors='coerce').fillna(12.0)
df['CLOSE_DEF_DIST'] = pd.to_numeric(df['CLOSE_DEF_DIST'], errors='coerce').fillna(4.0)
df['DRIBBLES'] = pd.to_numeric(df['DRIBBLES'], errors='coerce').fillna(0).astype(int)
df['TOUCH_TIME'] = pd.to_numeric(df['TOUCH_TIME'], errors='coerce').fillna(1.5)
df['FGM'] = pd.to_numeric(df['FGM'], errors='coerce').fillna(0).astype(int)
df['SHOT_DIST'] = pd.to_numeric(df['SHOT_DIST'], errors='coerce').fillna(15.0)
df['PTS_TYPE'] = pd.to_numeric(df['PTS_TYPE'], errors='coerce').fillna(2).astype(int)

# Рачунање каријерне заглађене прецизности (career_fg_smoothed)
LEAGUE_AVG = 0.45
WEIGHT = 25
player_stats = df.groupby('player_name').agg(
    career_total_shots=('FGM', 'count'),
    career_made_shots=('FGM', 'sum')
).reset_index()

player_stats['career_fg_pct'] = player_stats['career_made_shots'] / player_stats['career_total_shots'].replace(0, 1)
player_stats['career_fg_smoothed'] = (
    (player_stats['career_made_shots'] + WEIGHT * LEAGUE_AVG) / 
    (player_stats['career_total_shots'] + WEIGHT)
)

df = pd.merge(df, player_stats[['player_name', 'career_fg_smoothed', 'career_fg_pct', 'career_total_shots']], on='player_name', how='left')
df['career_fg_smoothed'] = df['career_fg_smoothed'].fillna(LEAGUE_AVG)

def format_defender_display(d_name):
    if pd.isna(d_name) or not str(d_name).strip():
        return "Сви дефанзивци (Просек лиге)"
    text = str(d_name).strip()
    if ',' in text:
        parts = [p.strip() for p in text.split(',')]
        if len(parts) >= 2:
            return f"{parts[1].title()} {parts[0].title()}"
    return text.title()

def normalize_defender_input(query):
    query = str(query).strip().lower()
    if not query or query in ['svi', 'all', '', 'сви']:
        return ""
    if ',' in query:
        parts = [p.strip() for p in query.split(',')]
        return f"{parts[0]}, {parts[1]}" if len(parts) >= 2 else query
    parts = query.split()
    if len(parts) >= 2:
        return f"{parts[-1]}, {' '.join(parts[:-1])}"
    return query

df['CLOSEST_DEFENDER_FORMATTED'] = df['CLOSEST_DEFENDER'].apply(format_defender_display)

# 10 карактеристика усклађених са моделом
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

# 2. Учитавање или тренирање модела
print("\n[2/4] Провера Random Forest модела (10 карактеристика)...")
ml_model = None
need_training = True

if os.path.exists(model_path):
    try:
        loaded = joblib.load(model_path)
        if hasattr(loaded, 'n_features_in_') and loaded.n_features_in_ == len(features):
            ml_model = loaded
            need_training = False
            print(" -> Учитан оптимизовани модел из 'models/nba_shot_model.joblib'!")
        else:
            print("  Непоклапање броја карактеристика, тренирам усаглашени модел...")
    except Exception as e:
        print(f"  Грешка при учитавању ({e}), покрећем ново тренирање...")

if need_training:
    print(" -> Тренирање Random Forest класификатора...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    ml_model = RandomForestClassifier(n_estimators=90, max_depth=10, min_samples_split=20, random_state=42, n_jobs=-1)
    ml_model.fit(X_train, y_train)
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(ml_model, model_path)
    print(" -> Модел је успешно снимљен.")

accuracy_val = round(accuracy_score(y, ml_model.predict(X)) * 100, 1)

# Падежне помоћне функције
def padez_godine(br):
    try:
        val = int(round(float(br)))
    except:
        return f"{br} година"
    last_two = val % 100
    last_one = val % 10
    if 11 <= last_two <= 14:
        return f"{val} година"
    if last_one == 1:
        return f"{val} година"
    if 2 <= last_one <= 4:
        return f"{val} године"
    return f"{val} година"

def padez_suteva(br):
    val = int(br)
    last_two = val % 100
    last_one = val % 10
    if 11 <= last_two <= 14:
        return f"{val:,} шутева"
    if last_one == 1:
        return f"{val:,} шут"
    if 2 <= last_one <= 4:
        return f"{val:,} шута"
    return f"{val:,} шутева"

# 3. Припрема играча
print("\n[3/4] Обрада играча и дефанзиваца...")
svi_igraci = {}

for name, group in df.groupby('player_name'):
    if len(group) < 8:
        continue

    total_fga = int(len(group))
    total_fgm = int(group['FGM'].sum())
    overall_pct = float(round((total_fgm / total_fga) * 100, 1))

    shots_2 = group[group['PTS_TYPE'] == 2]
    shots_3 = group[group['PTS_TYPE'] == 3]

    p2 = float(round((shots_2['FGM'].sum() / len(shots_2)) * 100, 1)) if len(shots_2) > 0 else 0.0
    p3 = float(round((shots_3['FGM'].sum() / len(shots_3)) * 100, 1)) if len(shots_3) > 0 else 0.0

    top_def = {}
    for d_name, d_group in group.groupby('CLOSEST_DEFENDER_FORMATTED'):
        if len(d_group) >= 2 and "Сви дефанзивци" not in d_name:
            top_def[str(d_name)] = {
                'made': int(d_group['FGM'].sum()),
                'total': int(len(d_group)),
                'pct': float(round((d_group['FGM'].sum() / len(d_group)) * 100, 1)),
                'dist': float(round(float(d_group['CLOSE_DEF_DIST'].mean()), 1))
            }

    career_smoothed = float(group['career_fg_smoothed'].iloc[0])
    exp_num = float(group['Experience_Num'].iloc[0])

    svi_igraci[name] = {
        'key': str(name),
        'display_name': str(name).title(),
        'pos': str(group['Pos'].iloc[0]) if 'Pos' in group.columns and pd.notna(group['Pos'].iloc[0]) else 'Бек / Крило',
        'height': str(group['Height'].iloc[0]) if 'Height' in group.columns and pd.notna(group['Height'].iloc[0]) else '6-6',
        'height_inches': float(group['Height_Inches'].iloc[0]),
        'exp': exp_num,
        'exp_str': padez_godine(exp_num),
        'career_fg_smoothed': float(round(career_smoothed, 4)),
        'total': total_fga,
        'total_str': padez_suteva(total_fga),
        'made': total_fgm,
        'overall_pct': overall_pct,
        'p2': p2,
        'p3': p3,
        'dist': float(round(float(group['SHOT_DIST'].mean()), 1)),
        'def_dist': float(round(float(group['CLOSE_DEF_DIST'].mean()), 1)),
        'dribbles': int(round(float(group['DRIBBLES'].mean()))),
        'touch': float(round(float(group['TOUCH_TIME'].mean()), 1)),
        'top_defenders': dict(sorted(top_def.items(), key=lambda x: x[1]['total'], reverse=True)[:12])
    }

svi_igraci_sortirani = dict(sorted(svi_igraci.items(), key=lambda x: x[1]['display_name']))

def convert_to_serializable(obj):
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    return obj

svi_igraci_json = json.dumps(convert_to_serializable(svi_igraci_sortirani), ensure_ascii=False)

# 4. Чист, прегледан HTML интерфејс (без слике и терена, са 2 јасне картице и ознаком 'сек')
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="sr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>НБА Аналитички Центар & Шут Симулатор</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=Oswald:wght@500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #070b14;
            --bg-card: #0c1222;
            --border: #1e293b;
            --accent: #ea580c;
            --cyan: #38bdf8;
            --red: #ef4444;
            --green: #10b981;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
        body { background: var(--bg-base); color: var(--text-main); min-height: 100vh; padding: 24px 20px; display: flex; justify-content: center; }
        .app-container { max-width: 1040px; width: 100%; display: flex; flex-direction: column; gap: 20px; }

        .header-bar {
            background: linear-gradient(135deg, #0d1527 0%, #131d36 50%, #1c1833 100%);
            border: 1px solid rgba(234, 88, 12, 0.25);
            border-radius: 16px;
            padding: 22px 28px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
            box-shadow: 0 16px 36px -10px rgba(0,0,0,0.7);
        }
        .header-left h1 {
            font-family: 'Oswald', sans-serif;
            font-size: 28px;
            letter-spacing: 0.8px;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 12px;
            text-transform: uppercase;
        }
        .header-left h1 span { color: var(--accent); }
        .header-left p { color: var(--text-muted); font-size: 13.5px; margin-top: 4px; }
        
        .header-badges { display: flex; gap: 10px; flex-wrap: wrap; }
        .pill-badge {
            background: rgba(11, 17, 32, 0.9);
            border: 1px solid #1e293b;
            padding: 8px 14px;
            border-radius: 10px;
            font-size: 12.5px;
            color: #cbd5e1;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .pill-badge b { color: var(--accent); }

        .workspace-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 768px) {
            .workspace-grid { grid-template-columns: 1fr; }
        }

        .panel {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 22px;
            display: flex;
            flex-direction: column;
            gap: 14px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.45);
        }
        .panel-title {
            font-size: 15px;
            font-weight: 800;
            color: #fff;
            display: flex;
            align-items: center;
            justify-content: space-between;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 12px;
            margin-bottom: 4px;
        }

        .form-group { margin-bottom: 8px; }
        label {
            display: block;
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }
        select, input {
            width: 100%;
            background: #080d18;
            border: 1px solid #223049;
            border-radius: 10px;
            padding: 11px 14px;
            color: #ffffff;
            font-size: 14px;
            font-weight: 500;
            outline: none;
            transition: all 0.2s;
        }
        select:focus, input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(234, 88, 12, 0.15);
        }
        input.invalid {
            border-color: var(--red);
            background: #1c0e12;
        }
        .err-hint {
            color: var(--red);
            font-size: 11px;
            margin-top: 4px;
            display: none;
            font-weight: 600;
        }

        .player-info-card {
            background: linear-gradient(135deg, #0e1726 0%, #131e33 100%);
            border: 1px solid #233352;
            border-radius: 12px;
            padding: 14px 16px;
        }
        .pic-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }
        .pic-title { font-size: 16px; font-weight: 800; color: var(--accent); }
        .pic-pos { font-size: 11.5px; color: #38bdf8; font-weight: 700; background: #082f49; padding: 2px 8px; border-radius: 5px; }
        .pic-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            margin-top: 10px;
            border-top: 1px solid #1e293b;
            padding-top: 10px;
        }
        .pic-stat-item { text-align: center; }
        .pic-stat-label { font-size: 10px; color: var(--text-muted); text-transform: uppercase; }
        .pic-stat-val { font-size: 14px; font-weight: 800; color: #fff; margin-top: 2px; }

        .h2h-card {
            background: #080d18;
            border: 1px solid #1e293b;
            border-left: 4px solid #38bdf8;
            border-radius: 10px;
            padding: 12px 14px;
            margin-top: 4px;
        }
        .h2h-headline { font-size: 11px; font-weight: 800; color: #38bdf8; text-transform: uppercase; margin-bottom: 4px; }
        .h2h-data { font-size: 14px; font-weight: 800; color: #fff; }
        .h2h-subtext { font-size: 11.5px; color: var(--text-muted); margin-top: 3px; line-height: 1.4; }

        .controls-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }

        .calc-btn {
            width: 100%;
            background: linear-gradient(135deg, #ea580c 0%, #c2410c 100%);
            border: none;
            color: #fff;
            padding: 15px;
            border-radius: 12px;
            font-size: 15px;
            font-weight: 800;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            cursor: pointer;
            box-shadow: 0 8px 24px rgba(234, 88, 12, 0.35);
            transition: all 0.2s;
            margin-top: 8px;
        }
        .calc-btn:hover {
            background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
            transform: translateY(-2px);
        }
        .calc-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        #result-wrapper { margin-top: 6px; display: none; }
        .result-card {
            padding: 24px;
            border-radius: 16px;
            text-align: center;
            border: 1px solid transparent;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }
        .res-success {
            background: linear-gradient(180deg, #063c1e 0%, #032311 100%);
            border-color: #10b981;
        }
        .res-fail {
            background: linear-gradient(180deg, #4c0d0d 0%, #2b0505 100%);
            border-color: #ef4444;
        }
        .res-header-badge { font-size: 18px; font-weight: 900; letter-spacing: 1px; }
        .res-prob-huge {
            font-family: 'Oswald', sans-serif;
            font-size: 58px;
            font-weight: 700;
            color: #fff;
            line-height: 1;
            margin: 8px 0;
        }
        .res-xp-pill {
            display: inline-block;
            background: rgba(0,0,0,0.4);
            border: 1px solid rgba(255,255,255,0.15);
            padding: 6px 18px;
            border-radius: 20px;
            font-size: 13.5px;
            font-weight: 700;
            color: #f8fafc;
        }
        .res-details-box {
            margin-top: 14px;
            background: rgba(0,0,0,0.3);
            padding: 14px;
            border-radius: 10px;
            font-size: 13.5px;
            line-height: 1.5;
            color: #cbd5e1;
            max-width: 650px;
            margin-left: auto;
            margin-right: auto;
        }
    </style>
</head>
<body>

<div class="app-container">
    <header class="header-bar">
        <div class="header-left">
            <h1>🏀 НБА Аналитички <span>Студио</span> & ML</h1>
            <p>Предикција успешности шута на бази 10 карактеристика и правила НБА игре</p>
        </div>
        <div class="header-badges">
            <div class="pill-badge">База: <b>__TOTAL_SHOTS__</b></div>
            <div class="pill-badge">Играча: <b>__TOTAL_PLAYERS__ у бази</b></div>
            <div class="pill-badge">Модел: <b style="color:#10b981">__ACCURACY__ тачност</b></div>
            <div class="pill-badge">Ограничење напада: <b>24 сек</b></div>
        </div>
    </header>

    <div class="workspace-grid">
        <!-- КАРТИЦА 1: ШУТЕР И ДЕФАНЗИВАЦ -->
        <div class="panel">
            <div class="panel-title">
                <span> 1. Шутер и Дефанзивац</span>
            </div>

            <div class="form-group">
                <label for="player_sel">Изаберите шутера (нападача):</label>
                <select id="player_sel" onchange="promeniIgraca()"></select>
            </div>

            <div class="player-info-card">
                <div class="pic-header">
                    <div class="pic-title" id="p_naziv">--</div>
                    <div class="pic-pos" id="p_pos">--</div>
                </div>
                <div style="font-size:12px; color:var(--text-muted);" id="p_dims">Висина: -- | Искуство: --</div>
                <div class="pic-stats">
                    <div class="pic-stat-item">
                        <div class="pic-stat-label">Узорак</div>
                        <div class="pic-stat-val" id="p_total">--</div>
                    </div>
                    <div class="pic-stat-item">
                        <div class="pic-stat-label">Шут за 2п</div>
                        <div class="pic-stat-val" id="p_p2" style="color:#ff9800;">--%</div>
                    </div>
                    <div class="pic-stat-item">
                        <div class="pic-stat-label">Шут за 3п</div>
                        <div class="pic-stat-val" id="p_p3" style="color:#38bdf8;">--%</div>
                    </div>
                </div>
            </div>

            <div class="form-group">
                <label for="defender_dropdown">🛡️ Дефанзивац (директни дуели):</label>
                <select id="defender_dropdown" onchange="izabranIzDropdowna()"></select>
            </div>

            <div class="form-group">
                <label for="custom_defender">Или упишите име било ког дефанзивца:</label>
                <input type="text" id="custom_defender" placeholder="нпр. Kawhi Leonard, Jimmy Butler..." oninput="proveriRucnogDefanzivca()">
            </div>

            <div class="h2h-card">
                <div class="h2h-headline">⚔️ Статистика овог дуела у бази:</div>
                <div class="h2h-data" id="h2h_stat">--</div>
                <div class="h2h-subtext" id="h2h_desc">--</div>
            </div>
        </div>

        <!-- КАРТИЦА 2: СИТУАЦИЈА ШУТА -->
        <div class="panel">
            <div class="panel-title">
                <span>🎯 2. Ситуација на паркету</span>
                <span id="pts_pill" style="font-size:11.5px; font-weight:800; background:#ea580c; color:#fff; padding:3px 10px; border-radius:6px;">Шут за 3 поена</span>
            </div>

            <div class="controls-grid">
                <div>
                    <label for="dist">Дистанца од коша (ft) [≥ 0]:</label>
                    <input type="number" id="dist" step="0.5" min="0" max="94" value="24.0" oninput="inputChanged()">
                    <div id="dist_err" class="err-hint">Дистанца мора бити позитивна (0-94 ft)!</div>
                </div>
                <div>
                    <label for="def_dist">Близина одбране (ft) [≥ 0]:</label>
                    <input type="number" id="def_dist" step="0.5" min="0" max="50" value="4.0" oninput="inputChanged()">
                    <div id="def_dist_err" class="err-hint">Близина не сме бити негативна!</div>
                </div>
                <div>
                    <label for="clock">Сат напада [0.1 - 24.0 сек]:</label>
                    <input type="number" id="clock" step="0.1" min="0.1" max="24.0" value="12.0" oninput="inputChanged()">
                    <div id="clock_err" class="err-hint">НБА сат је тачно од 0.1 до 24.0 сек!</div>
                </div>
                <div>
                    <label for="dribbles">Број дриблинга [≥ 0]:</label>
                    <input type="number" id="dribbles" min="0" max="50" step="1" value="1" oninput="inputChanged()">
                    <div id="dribbles_err" class="err-hint">Број дриблинга не може бити негативан!</div>
                </div>
                <div>
                    <label for="touch">Време са лоптом (сек) [≥ 0]:</label>
                    <input type="number" id="touch" step="0.1" min="0" max="24.0" value="1.5" oninput="inputChanged()">
                    <div id="touch_err" class="err-hint">Време не може бити негативно (максимално 24 сек)!</div>
                </div>
                <div>
                    <label for="is_home">Терен утакмице:</label>
                    <select id="is_home">
                        <option value="1">Домаћи терен 🏠</option>
                        <option value="0">Гостујући терен ✈️</option>
                    </select>
                </div>
            </div>

            <button class="calc-btn" id="calc_btn" onclick="pokreniPredikciju()">⚡ ИЗРАЧУНАЈ ШАНСУ ЗА КОШ</button>
        </div>
    </div>

    <!-- РЕЗУЛТАТ -->
    <div id="result-wrapper">
        <div id="res_card" class="result-card">
            <div id="res_badge" class="res-header-badge">--</div>
            <div id="res_prob" class="res-prob-huge">--%</div>
            <div id="res_xp" class="res-xp-pill">Очекивани поени (xP): --</div>
            <div id="res_desc" class="res-details-box">--</div>
        </div>
    </div>
</div>

<script>
let baza = {};
const playerSelect = document.getElementById('player_sel');
const defDropdown = document.getElementById('defender_dropdown');
const customDefInput = document.getElementById('custom_defender');
const calcBtn = document.getElementById('calc_btn');

let currentHeight = 78;
let currentExp = 4;
let currentCareerSmoothed = 0.45;

function promeniIgraca() {
    const k = playerSelect.value;
    const igrac = baza[k];
    if (!igrac) return;

    document.getElementById('p_naziv').innerText = igrac.display_name;
    document.getElementById('p_pos').innerText = igrac.pos;
    document.getElementById('p_dims').innerText = 'Висина: ' + igrac.height + ' | Искуство: ' + igrac.exp_str;
    document.getElementById('p_total').innerText = igrac.total_str;
    document.getElementById('p_p2').innerText = igrac.p2 + '%';
    document.getElementById('p_p3').innerText = igrac.p3 + '%';

    document.getElementById('dist').value = igrac.dist;
    document.getElementById('dribbles').value = igrac.dribbles;
    document.getElementById('touch').value = igrac.touch;

    currentHeight = igrac.height_inches;
    currentExp = igrac.exp;
    currentCareerSmoothed = igrac.career_fg_smoothed;

    defDropdown.innerHTML = '';
    const optSvi = document.createElement('option');
    optSvi.value = '__svi__';
    optSvi.innerText = '🛡️ Сви дефанзивци (Просек лиге)';
    defDropdown.appendChild(optSvi);

    for (const [defName, dStat] of Object.entries(igrac.top_defenders)) {
        const opt = document.createElement('option');
        opt.value = defName;
        opt.innerText = '🛡️ ' + defName + ' (' + dStat.total + ' дуела)';
        defDropdown.appendChild(opt);
    }

    customDefInput.value = '';
    izabranIzDropdowna();
}

function izabranIzDropdowna() {
    const k = playerSelect.value;
    const igrac = baza[k];
    if (!igrac) return;
    const defVal = defDropdown.value;

    if (defVal === '__svi__') {
        document.getElementById('def_dist').value = igrac.def_dist;
        document.getElementById('h2h_stat').innerText = 'Просек против свих: 2п ' + igrac.p2 + '% | 3п ' + igrac.p3 + '%';
        document.getElementById('h2h_desc').innerText = 'Просечна удаљеност чувара у сезони износи ' + igrac.def_dist + ' ft.';
    } else {
        customDefInput.value = defVal;
        const dStat = igrac.top_defenders[defVal];
        if (dStat) {
            document.getElementById('def_dist').value = dStat.dist;
            document.getElementById('h2h_stat').innerText = 'Погодио ' + dStat.made + ' од ' + dStat.total + ' шутева (' + dStat.pct + '%)';
            document.getElementById('h2h_desc').innerText = defVal + ' га у просеку држи на ' + dStat.dist + ' ft удаљености.';
        }
    }
    inputChanged();
    pokreniPredikciju();
}

function proveriRucnogDefanzivca() {
    const ime = customDefInput.value.trim();
    if (!ime) {
        defDropdown.value = '__svi__';
        izabranIzDropdowna();
        return;
    }

    fetch('/check_defender', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ player_key: playerSelect.value, defender_name: ime })
    })
    .then(r => r.json())
    .then(data => {
        if (data.pronadjen) {
            document.getElementById('h2h_stat').innerText = 'Погодио ' + data.made + ' од ' + data.total + ' шутева (' + data.pct + '%)';
            document.getElementById('h2h_desc').innerText = data.ime + ' га у просеку држи на ' + data.dist + ' ft удаљености.';
            document.getElementById('def_dist').value = data.dist;
        } else {
            document.getElementById('h2h_stat').innerText = 'Нема директних дуела у бази са играчем: ' + ime;
            document.getElementById('h2h_desc').innerText = 'Модел примењује стандардну одбрану са унетом удаљеношћу.';
        }
        inputChanged();
    });
}

function inputChanged() {
    let valid = true;

    const distEl = document.getElementById('dist');
    const defDistEl = document.getElementById('def_dist');
    const clockEl = document.getElementById('clock');
    const dribblesEl = document.getElementById('dribbles');
    const touchEl = document.getElementById('touch');

    const dist = parseFloat(distEl.value);
    const defDist = parseFloat(defDistEl.value);
    const clock = parseFloat(clockEl.value);
    const dribbles = parseInt(dribblesEl.value);
    const touch = parseFloat(touchEl.value);

    if (isNaN(dist) || dist < 0 || dist > 94) {
        distEl.classList.add('invalid');
        document.getElementById('dist_err').style.display = 'block';
        valid = false;
    } else {
        distEl.classList.remove('invalid');
        document.getElementById('dist_err').style.display = 'none';
    }

    if (isNaN(defDist) || defDist < 0) {
        defDistEl.classList.add('invalid');
        document.getElementById('def_dist_err').style.display = 'block';
        valid = false;
    } else {
        defDistEl.classList.remove('invalid');
        document.getElementById('def_dist_err').style.display = 'none';
    }

    if (isNaN(clock) || clock < 0.1 || clock > 24.0) {
        clockEl.classList.add('invalid');
        document.getElementById('clock_err').style.display = 'block';
        valid = false;
    } else {
        clockEl.classList.remove('invalid');
        document.getElementById('clock_err').style.display = 'none';
    }

    if (isNaN(dribbles) || dribbles < 0) {
        dribblesEl.classList.add('invalid');
        document.getElementById('dribbles_err').style.display = 'block';
        valid = false;
    } else {
        dribblesEl.classList.remove('invalid');
        document.getElementById('dribbles_err').style.display = 'none';
    }

    if (isNaN(touch) || touch < 0 || touch > 24.0) {
        touchEl.classList.add('invalid');
        document.getElementById('touch_err').style.display = 'block';
        valid = false;
    } else {
        touchEl.classList.remove('invalid');
        document.getElementById('touch_err').style.display = 'none';
    }

    const pts = (!isNaN(dist) && dist >= 22.5) ? 3 : 2;
    const ptsPill = document.getElementById('pts_pill');
    if (pts === 3) {
        ptsPill.innerText = 'Шут за 3 поена (≥ 22.5 ft)';
        ptsPill.style.background = '#0284c7';
    } else {
        ptsPill.innerText = 'Шут за 2 поена (< 22.5 ft)';
        ptsPill.style.background = '#ea580c';
    }

    calcBtn.disabled = !valid;
    return valid;
}

function pokreniPredikciju() {
    if (!inputChanged()) return;

    const dist = Math.max(0, parseFloat(document.getElementById('dist').value));
    const pts = dist >= 22.5 ? 3 : 2;
    const defIme = customDefInput.value.trim() || 'просечне одбране';

    const payload = {
        dist: dist,
        def_dist: Math.max(0, parseFloat(document.getElementById('def_dist').value)),
        clock: Math.min(24.0, Math.max(0.1, parseFloat(document.getElementById('clock').value))),
        dribbles: Math.max(0, parseInt(document.getElementById('dribbles').value)),
        touch: Math.max(0, parseFloat(document.getElementById('touch').value)),
        is_home: parseInt(document.getElementById('is_home').value),
        height: currentHeight,
        exp: currentExp,
        career_fg_smoothed: currentCareerSmoothed,
        defender: defIme,
        pts: pts
    };

    fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
    })
    .then(r => r.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }

        document.getElementById('result-wrapper').style.display = 'block';
        const card = document.getElementById('res_card');
        const badge = document.getElementById('res_badge');

        if (data.ishod === 'ПОГОДАК') {
            card.className = 'result-card res-success';
            badge.innerText = 'ПРЕДИКЦИЈА МОДЕЛА: ПОГОДАК';
            badge.style.color = '#34d399';
        } else {
            card.className = 'result-card res-fail';
            badge.innerText = 'ПРЕДИКЦИЈА МОДЕЛА: ПРОМАШАЈ';
            badge.style.color = '#f87171';
        }

        document.getElementById('res_prob').innerText = data.prob + '%';
        document.getElementById('res_xp').innerText = data.xp_tekst;
        document.getElementById('res_desc').innerText = data.komentar;
    });
}

fetch('/get_players')
    .then(r => r.json())
    .then(data => {
        baza = data;
        playerSelect.innerHTML = '';
        let foundPreferred = false;
        for (const [key, p] of Object.entries(baza)) {
            const opt = document.createElement('option');
            opt.value = key;
            opt.innerText = p.display_name;
            if (!foundPreferred && (key.includes('lebron') || key.includes('curry') || key.includes('kobe'))) {
                opt.selected = true;
                foundPreferred = true;
            }
            playerSelect.appendChild(opt);
        }
        promeniIgraca();
    });
</script>

</body>
</html>
"""

HTML_KOD = (
    HTML_TEMPLATE
    .replace('__TOTAL_SHOTS__', f"{len(ml_data):,}")
    .replace('__TOTAL_PLAYERS__', f"{len(svi_igraci_sortirani)}")
    .replace('__ACCURACY__', f"{accuracy_val}%")
)

class ReusableServer(socketserver.TCPServer):
    allow_reuse_address = True

class MLHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path in ['/', '/index.html']:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_KOD.encode('utf-8'))
        elif self.path == '/get_players':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(svi_igraci_json.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get('content-length', 0))
        body = self.rfile.read(length)
        req = json.loads(body.decode('utf-8'))

        if self.path == '/check_defender':
            p_key = req.get('player_key', '')
            d_name_raw = req.get('defender_name', '')
            d_kljuc = normalize_defender_input(d_name_raw)

            sub = df[(df['player_name'] == p_key) & (df['CLOSEST_DEFENDER'].astype(str).str.lower() == d_kljuc)]
            if len(sub) > 0:
                made = int(sub['FGM'].sum())
                total = int(len(sub))
                pct = float(round((made / total) * 100, 1))
                dist = float(round(float(sub['CLOSE_DEF_DIST'].mean()), 1))
                res = {'pronadjen': True, 'ime': format_defender_display(d_kljuc), 'made': made, 'total': total, 'pct': pct, 'dist': dist}
            else:
                res = {'pronadjen': False}

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(res, ensure_ascii=False).encode('utf-8'))

        elif self.path == '/predict':
            try:
                shot_dist = max(0.0, float(req['dist']))
                close_def = max(0.0, float(req['def_dist']))
                shot_clock = float(req['clock'])
                
                # Провера НБА правила: сат искључиво од 0.1 до 24.0 сек
                if shot_clock < 0.1 or shot_clock > 24.0:
                    raise ValueError("Сат напада мора бити између 0.1 и 24.0 сек према правилима НБА!")

                dribbles = max(0, int(req['dribbles']))
                touch_time = max(0.0, min(24.0, float(req['touch'])))
                is_home = 1 if int(req.get('is_home', 1)) == 1 else 0
                height_in = max(0.0, float(req.get('height', 78.0)))
                exp_num = max(0.0, float(req.get('exp', 4.0)))
                career_smoothed = float(req.get('career_fg_smoothed', LEAGUE_AVG))

                pts_type = int(req.get('pts', 2))
                if pts_type < 0:
                    pts_type = 2 if shot_dist < 22.5 else 3

                # Улазни вектор са свих 10 карактеристика
                df_input = pd.DataFrame([{
                    'SHOT_DIST': shot_dist,
                    'CLOSE_DEF_DIST': close_def,
                    'SHOT_CLOCK': shot_clock,
                    'DRIBBLES': dribbles,
                    'TOUCH_TIME': touch_time,
                    'PTS_TYPE': pts_type,
                    'Is_Home': is_home,
                    'Height_Inches': height_in,
                    'Experience_Num': exp_num,
                    'career_fg_smoothed': career_smoothed
                }])

                prob = float(ml_model.predict_proba(df_input)[0][1])
                ishod = "ПОГОДАК" if prob >= 0.50 else "ПРОМАШАЈ"
                xp = max(0.0, round(prob * pts_type, 2))
                def_ime = req.get('defender', 'дефанзивца')

                # Падеж за поене
                xp_val_int = int(xp)
                xp_nastavak = "поена"
                if xp == 1.0 or (xp_val_int % 10 == 1 and xp_val_int % 100 != 11 and xp == xp_val_int):
                    xp_nastavak = "поен"
                elif 2 <= (xp_val_int % 10) <= 4 and not (12 <= (xp_val_int % 100) <= 14) and xp == xp_val_int:
                    xp_nastavak = "поена"

                xp_tekst = f"Очекивани поени из шута (xP): {xp:.2f} {xp_nastavak}"

                if prob >= 0.50:
                    komentar = f"Повољна позиција за шут! Шутер када га чува {def_ime} на удаљености од {close_def} ft уз {shot_clock:.1f} сек на сату напада има вероватноћу поготка од {prob*100:.1f}%. Модел предвиђа ПОГОДАК за {pts_type} поена."
                else:
                    komentar = f"Тежак шут под дефанзивним притиском! Дефанзивац ({def_ime}) на {close_def} ft врши јак притисак уз {shot_clock:.1f} сек до истека напада. Модел предвиђа ПРОМАШАЈ (вероватноћа {prob*100:.1f}%)."

                res = {
                    'prob': f"{prob * 100:.1f}",
                    'ishod': ishod,
                    'xp': f"{xp:.2f}",
                    'xp_tekst': xp_tekst,
                    'komentar': komentar
                }
            except Exception as ex:
                res = {'error': f"Грешка: {str(ex)}"}

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(res, ensure_ascii=False).encode('utf-8'))

PORT = 5050
print("==================================================")
print("🚀 [4/4] ПОКРЕЋЕМ НБА АНАЛИТИЧКИ СТУДИО...")
print("==================================================")
print(f"\n👉 ОТВОРИ У БРАУЗЕРУ: http://127.0.0.1:{PORT}\n")

try:
    subprocess.Popen(['xdg-open', f'http://127.0.0.1:{PORT}'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except:
    try:
        webbrowser.open(f'http://127.0.0.1:{PORT}')
    except:
        pass

server = ReusableServer(('127.0.0.1', PORT), MLHandler)
try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\nСервер је заустављен.")