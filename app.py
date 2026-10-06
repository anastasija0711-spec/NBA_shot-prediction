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


# Učitavanje baze
curr_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(curr_dir) == 'models':
    root_dir = os.path.dirname(curr_dir)
else:
    root_dir = curr_dir

shots_path = os.path.join(root_dir, 'data', 'shot_logs.csv')
players_path = os.path.join(root_dir, 'data', 'players.csv')
model_path = os.path.join(root_dir, 'models', 'nba_shot_model.joblib')

if not os.path.exists(shots_path) or not os.path.exists(players_path):
    print("❌ Greška: 'shot_logs.csv' i 'players.csv' moraju biti u folderu 'data/'.")
    sys.exit(1)

print("🏀 Učitavam baze šuteva i defanzivaca...")
shots_df = pd.read_csv(shots_path)
players_df = pd.read_csv(players_path)

shots_df['player_name'] = shots_df['player_name'].astype(str).str.lower().str.strip()
players_df['Name'] = players_df['Name'].astype(str).str.lower().str.strip()
players_clean = players_df.drop_duplicates(subset=['Name'], keep='last')

zeljene_kolone = [k for k in ['Name', 'Pos', 'Height', 'Experience'] if k in players_clean.columns]
players_clean = players_clean[zeljene_kolone]

df = pd.merge(shots_df, players_clean, left_on='player_name', right_on='Name', how='left')

def height_to_inches(h_str):
    try:
        parts = str(h_str).split('-')
        if len(parts) == 2:
            return int(parts[0]) * 12 + int(parts[1])
        return 78.0
    except:
        return 78.0

df['Height_Inches'] = df['Height'].apply(height_to_inches)
df['Experience_Num'] = pd.to_numeric(df['Experience'], errors='coerce').fillna(4.0)
df['Is_Home'] = (df['LOCATION'].astype(str).str.upper() == 'H').astype(int)
df['SHOT_CLOCK'] = df['SHOT_CLOCK'].fillna(12.0)
df['CLOSE_DEF_DIST'] = df['CLOSE_DEF_DIST'].fillna(4.0)
df['DRIBBLES'] = pd.to_numeric(df['DRIBBLES'], errors='coerce').fillna(0)
df['TOUCH_TIME'] = pd.to_numeric(df['TOUCH_TIME'], errors='coerce').fillna(1.5)
df['FGM'] = pd.to_numeric(df['FGM'], errors='coerce').fillna(0).astype(int)

# Funkcije za normalizaciju defanzivaca
def formatiraj_za_prikaz(ime_baze):
    if pd.isna(ime_baze) or not str(ime_baze).strip():
        return "Svi defanzivci"
    tekst = str(ime_baze).strip()
    if ',' in tekst:
        d = [p.strip() for p in tekst.split(',')]
        if len(d) >= 2:
            return f"{d[1].title()} {d[0].title()}"
    return tekst.title()

def normalizuj_unos_defanzivca(unos):
    unos = str(unos).strip().lower()
    if not unos or unos in ['svi', 'all', '']:
        return ""
    if ',' in unos:
        d = [p.strip() for p in unos.split(',')]
        return f"{d[0]}, {d[1]}" if len(d) >= 2 else unos
    d = unos.split()
    if len(d) >= 2:
        return f"{d[-1]}, {' '.join(d[:-1])}"
    return unos

df['CLOSEST_DEFENDER_FORMATTED'] = df['CLOSEST_DEFENDER'].apply(formatiraj_za_prikaz)

features = [
    'SHOT_DIST', 'CLOSE_DEF_DIST', 'SHOT_CLOCK', 
    'DRIBBLES', 'TOUCH_TIME', 'PTS_TYPE', 
    'Is_Home', 'Height_Inches', 'Experience_Num'
]
ml_data = df.dropna(subset=features + ['FGM'])
X = ml_data[features]
y = ml_data['FGM']


# Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

if os.path.exists(model_path):
    try:
        ml_model = joblib.load(model_path)
    except:
        ml_model = RandomForestClassifier(n_estimators=70, max_depth=10, random_state=42, n_jobs=-1)
        ml_model.fit(X_train, y_train)
        os.makedirs(os.path.join(root_dir, 'models'), exist_ok=True)
        joblib.dump(ml_model, model_path)
else:
    print("Treniram model...")
    ml_model = RandomForestClassifier(n_estimators=70, max_depth=10, random_state=42, n_jobs=-1)
    ml_model.fit(X_train, y_train)
    os.makedirs(os.path.join(root_dir, 'models'), exist_ok=True)
    joblib.dump(ml_model, model_path)

accuracy_val = round(accuracy_score(y_test, ml_model.predict(X_test)) * 100, 1)

# Priprema igrača i njihovih defanzivaca
svi_igraci = {}
for name, group in df.groupby('player_name'):
    if len(group) < 10:
        continue
    shots_2 = group[group['PTS_TYPE'] == 2]
    p2 = round((shots_2['FGM'].sum() / len(shots_2)) * 100, 1) if len(shots_2) > 0 else 0.0
    shots_3 = group[group['PTS_TYPE'] == 3]
    p3 = round((shots_3['FGM'].sum() / len(shots_3)) * 100, 1) if len(shots_3) > 0 else 0.0

    top_def = {}
    for d_name, d_group in group.groupby('CLOSEST_DEFENDER_FORMATTED'):
        if len(d_group) >= 2 and d_name != "Svi Defanzivci":
            top_def[d_name] = {
                'made': int(d_group['FGM'].sum()),
                'total': int(len(d_group)),
                'pct': round((d_group['FGM'].sum() / len(d_group)) * 100, 1),
                'dist': round(float(d_group['CLOSE_DEF_DIST'].mean()), 1)
            }

    svi_igraci[name] = {
        'display_name': name.title(),
        'pos': str(group['Pos'].iloc[0]) if 'Pos' in group.columns and pd.notna(group['Pos'].iloc[0]) else 'N/A',
        'height': str(group['Height'].iloc[0]) if 'Height' in group.columns and pd.notna(group['Height'].iloc[0]) else '6-6',
        'height_inches': float(group['Height_Inches'].iloc[0]),
        'exp': float(group['Experience_Num'].iloc[0]),
        'total': int(len(group)),
        'p2': p2,
        'p3': p3,
        'dist': round(float(group['SHOT_DIST'].mean()), 1),
        'def_dist': round(float(group['CLOSE_DEF_DIST'].mean()), 1),
        'dribbles': int(round(float(group['DRIBBLES'].mean()))),
        'touch': round(float(group['TOUCH_TIME'].mean()), 1),
        'top_defenders': dict(sorted(top_def.items(), key=lambda x: x[1]['total'], reverse=True)[:10])
    }

svi_igraci_sortirani = dict(sorted(svi_igraci.items(), key=lambda x: x[1]['display_name']))


HTML_KOD = f"""<!DOCTYPE html>
<html lang="sr">
<head>
    <meta charset="UTF-8">
    <title>NBA Machine Learning Prediktor</title>
    <style>
        * {{ box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        body {{ background: #0b1120; color: #f8fafc; margin: 0; padding: 24px; display: flex; justify-content: center; }}
        .app {{ max-width: 960px; width: 100%; }}
        
        .header {{ background: #131c2e; border: 1px solid #1e293b; border-radius: 14px; padding: 20px 24px; margin-bottom: 20px; }}
        .header h1 {{ margin: 0 0 6px 0; color: #ea580c; font-size: 24px; }}
        .header p {{ margin: 0; color: #94a3b8; font-size: 14px; }}

        .badge-row {{ display: flex; gap: 10px; margin-top: 12px; }}
        .badge {{ background: #0b1120; border: 1px solid #334155; padding: 6px 12px; border-radius: 6px; font-size: 13px; color: #cbd5e1; }}
        .badge b {{ color: #ea580c; }}

        .grid-cards {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
        .card {{ background: #131c2e; border: 1px solid #1e293b; border-radius: 14px; padding: 22px; }}
        .title {{ font-size: 15px; font-weight: bold; color: #f8fafc; margin-top: 0; margin-bottom: 14px; }}

        label {{ display: block; font-size: 12px; font-weight: 600; color: #94a3b8; margin-bottom: 5px; }}
        input, select {{ width: 100%; background: #0b1120; border: 1px solid #334155; border-radius: 8px; padding: 10px 12px; color: white; font-size: 14px; outline: none; margin-bottom: 12px; }}
        input:focus, select:focus {{ border-color: #ea580c; }}

        .h2h-box {{ background: #0b1120; border: 1px solid #334155; border-radius: 10px; padding: 14px; margin-top: 8px; }}
        .h2h-title {{ font-size: 12px; font-weight: bold; color: #38bdf8; text-transform: uppercase; margin-bottom: 4px; }}
        .h2h-stat {{ font-size: 15px; font-weight: bold; color: white; }}
        .h2h-sub {{ font-size: 12px; color: #94a3b8; margin-top: 2px; }}

        .form-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}

        .btn {{ width: 100%; background: #ea580c; color: white; border: none; padding: 15px; border-radius: 10px; font-size: 16px; font-weight: bold; cursor: pointer; transition: 0.2s; box-shadow: 0 4px 14px rgba(234, 88, 12, 0.4); margin-top: 10px; }}
        .btn:hover {{ background: #c2410c; }}

        #result-box {{ margin-top: 20px; display: none; }}
        .res-container {{ padding: 22px; border-radius: 14px; text-align: center; border: 1px solid transparent; }}
        .res-pogodak {{ background: #14532d; border-color: #22c55e; }}
        .res-promasaj {{ background: #7f1d1d; border-color: #ef4444; }}
        .res-status {{ font-size: 24px; font-weight: 800; margin-bottom: 6px; }}
        .res-prob {{ font-size: 40px; font-weight: 900; margin: 4px 0; color: white; }}
        .res-xp {{ font-size: 15px; color: #e2e8f0; font-weight: 600; }}
        .res-comment {{ font-size: 14px; margin-top: 12px; background: rgba(0,0,0,0.25); padding: 12px; border-radius: 8px; }}
    </style>
</head>
<body>

<div class="app">
    <div class="header">
        <h1>🏀 NBA Machine Learning: Šuter i Njegov Defanzivac</h1>
        <p>Predikcija uspešnosti šuta kada igrač ima svog konkretnog defanzivca ili igra protiv prosečne odbrane.</p>
        <div class="badge-row">
            <div class="badge">Baza: <b>{len(ml_data):,} šuteva</b></div>
            <div class="badge">Igrača: <b>{len(svi_igraci_sortirani)}</b></div>
            <div class="badge">Tačnost modela: <b style="color:#4ade80">{accuracy_val}%</b></div>
        </div>
    </div>

    <div class="grid-cards">
        <!-- LEVO: ŠUTER I DEFANZIVAC -->
        <div class="card">
            <div class="title">👤 1. Šuter i Defanzivac</div>

            <label>Izaberi šutera (napadača):</label>
            <select id="player_sel" onchange="promeniIgraca()">
            </select>

            <div style="background:#0b1120; padding:12px; border-radius:8px; margin-bottom:14px; border:1px solid #1e293b;">
                <div id="p_naziv" style="font-size:16px; font-weight:bold; color:#ea580c;">--</div>
                <div id="p_opis" style="font-size:12px; color:#94a3b8; margin:2px 0;">--</div>
                <div id="p_stats" style="font-size:13px; font-weight:bold; color:white;">--</div>
            </div>

            <label>🛡️ Defanzivac (izaberi iz liste ili upiši svog):</label>
            <select id="defender_dropdown" onchange="izabranIzDropdowna()">
            </select>

            <label style="margin-top:4px;">Ili ručno upiši ime bilo kog defanzivca:</label>
            <input type="text" id="custom_defender" placeholder="npr. Jimmy Butler ili Kawhi Leonard" oninput="proveriRucnogDefanzivca()">

            <div id="h2h_box" class="h2h-box">
                <div class="h2h-title">⚔️ Statistika ovog direktnog duela u bazi:</div>
                <div id="h2h_stat" class="h2h-stat">--</div>
                <div id="h2h_desc" class="h2h-sub">--</div>
            </div>
        </div>

        <!-- DESNO: PARAMETRI ŠUTA -->
        <div class="card">
            <div class="title">🎯 2. Situacija šuta na parketu</div>
            <div class="form-grid">
                <div>
                    <label>Udaljenost od koša (ft):</label>
                    <input type="number" id="dist" step="0.5" value="24.0">
                </div>
                <div>
                    <label>Blizina defanzivca (ft):</label>
                    <input type="number" id="def_dist" step="0.5" value="4.0">
                </div>
                <div>
                    <label>Sat napada (1-24s):</label>
                    <input type="number" id="clock" step="0.5" value="12.0">
                </div>
                <div>
                    <label>Broj driblinga:</label>
                    <input type="number" id="dribbles" value="1">
                </div>
                <div>
                    <label>Vreme sa loptom (s):</label>
                    <input type="number" id="touch" step="0.1" value="1.5">
                </div>
                <div>
                    <label>Teren utakmice:</label>
                    <select id="is_home" style="margin-bottom:0">
                        <option value="1">Domaći teren 🏠</option>
                        <option value="0">Gostujući teren ✈️</option>
                    </select>
                </div>
            </div>

            <button class="btn" onclick="pokreniPredikciju()">⚡ IZRAČUNAJ ŠANSU ZA KOŠ</button>
        </div>
    </div>

    <!-- REZULTAT -->
    <div id="result-box">
        <div id="res_container" class="res-container">
            <div id="res_status" class="res-status">--</div>
            <div id="res_prob" class="res-prob">--%</div>
            <div id="res_xp" class="res-xp">--</div>
            <div id="res_comment" class="res-comment">--</div>
        </div>
    </div>
</div>

<script>
const baza = {json.dumps(svi_igraci_sortirani)};
const playerSelect = document.getElementById('player_sel');
const defDropdown = document.getElementById('defender_dropdown');
const customDefInput = document.getElementById('custom_defender');

for (const [key, p] of Object.entries(baza)) {{
    const opt = document.createElement('option');
    opt.value = key;
    opt.innerText = p.display_name;
    if (key === 'lebron james') opt.selected = true;
    playerSelect.appendChild(opt);
}}

let visinaInchi = 81;
let godineIskustva = 12;

function promeniIgraca() {{
    const k = playerSelect.value;
    const igrac = baza[k];
    if (!igrac) return;

    document.getElementById('p_naziv').innerText = igrac.display_name;
    document.getElementById('p_opis').innerText = `Pozicija: ${{igrac.pos}} | Visina: ${{igrac.height}} | Iskustvo: ${{igrac.exp}} god.`;
    document.getElementById('p_stats').innerText = `Sezona: 2p: ${{igrac.p2}}% | 3p: ${{igrac.p3}}% (Ukupno: ${{igrac.total}} šuteva)`;

    document.getElementById('dist').value = igrac.dist;
    document.getElementById('dribbles').value = igrac.dribbles;
    document.getElementById('touch').value = igrac.touch;
    visinaInchi = igrac.height_inches;
    godineIskustva = igrac.exp;

    // Popunjavanje dropdowna defanzivaca
    defDropdown.innerHTML = '';
    const optSvi = document.createElement('option');
    optSvi.value = '__svi__';
    optSvi.innerText = '🛡️ Svi defanzivci (Prosek lige)';
    defDropdown.appendChild(optSvi);

    for (const [defName, dStat] of Object.entries(igrac.top_defenders)) {{
        const opt = document.createElement('option');
        opt.value = defName;
        opt.innerText = `🛡️ ${{defName}} (${{dStat.total}} duela u sezoni)`;
        defDropdown.appendChild(opt);
    }}

    customDefInput.value = '';
    izabranIzDropdowna();
}}

function izabranIzDropdowna() {{
    const k = playerSelect.value;
    const igrac = baza[k];
    const defVal = defDropdown.value;

    if (defVal === '__svi__') {{
        document.getElementById('def_dist').value = igrac.def_dist;
        document.getElementById('h2h_stat').innerText = `Prosek protiv svih: 2p ${{igrac.p2}}% | 3p ${{igrac.p3}}%`;
        document.getElementById('h2h_desc').innerText = `Prosečna udaljenost odbrambenih igrača: ${{igrac.def_dist}} ft.`;
    }} else {{
        customDefInput.value = defVal;
        const dStat = igrac.top_defenders[defVal];
        if (dStat) {{
            document.getElementById('def_dist').value = dStat.dist;
            document.getElementById('h2h_stat').innerText = `Pogodio ${{dStat.made}} od ${{dStat.total}} šuteva (${{dStat.pct}}%)`;
            document.getElementById('h2h_desc').innerText = `${{defVal}} ga u proseku pritiska na ${{dStat.dist}} ft udaljenosti.`;
        }}
    }}
    pokreniPredikciju();
}}

function proveriRucnogDefanzivca() {{
    const ime = customDefInput.value.trim();
    if (!ime) {{
        defDropdown.value = '__svi__';
        izabranIzDropdowna();
        return;
    }}

    fetch('/check_defender', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{ player_key: playerSelect.value, defender_name: ime }})
    }})
    .then(r => r.json())
    .then(data => {{
        if (data.pronadjen) {{
            document.getElementById('h2h_stat').innerText = `Pogodio ${{data.made}} od ${{data.total}} šuteva (${{data.pct}}%)`;
            document.getElementById('h2h_desc').innerText = `${{data.ime}} ga u proseku drži na ${{data.dist}} ft udaljenosti.`;
            document.getElementById('def_dist').value = data.dist;
        }} else {{
            document.getElementById('h2h_stat').innerText = `Nema direktnih duela u bazi sa: ${{ime}}`;
            document.getElementById('h2h_desc').innerText = `Model će primeniti standardnu odbranu za ovog defanzivca.`;
        }}
    }});
}}

function pokreniPredikciju() {{
    const dist = parseFloat(document.getElementById('dist').value);
    const pts = dist >= 22.5 ? 3 : 2;
    const defIme = customDefInput.value.trim() || 'prosečne odbrane';

    const payload = {{
        dist: dist,
        def_dist: parseFloat(document.getElementById('def_dist').value),
        clock: parseFloat(document.getElementById('clock').value),
        dribbles: parseInt(document.getElementById('dribbles').value),
        touch: parseFloat(document.getElementById('touch').value),
        is_home: parseInt(document.getElementById('is_home').value),
        height: visinaInchi,
        exp: godineIskustva,
        defender: defIme,
        pts: pts
    }};

    fetch('/predict', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(payload)
    }})
    .then(r => r.json())
    .then(data => {{
        document.getElementById('result-box').style.display = 'block';
        const box = document.getElementById('res_container');
        const statusEl = document.getElementById('res_status');
        
        if (data.ishod === 'POGODAK') {{
            box.className = 'res-container res-pogodak';
            statusEl.innerText = '🟢 PREDIKCIJA MODELA: POGODAK';
            statusEl.style.color = '#4ade80';
        }} else {{
            box.className = 'res-container res-promasaj';
            statusEl.innerText = '🔴 PREDIKCIJA MODELA: PROMAŠAJ';
            statusEl.style.color = '#f87171';
        }}

        document.getElementById('res_prob').innerText = data.prob + '%';
        document.getElementById('res_xp').innerText = 'Očekivani poeni iz ovog šuta (xP): ' + data.xp + ' pts';
        document.getElementById('res_comment').innerText = data.komentar;
    }});
}}

promeniIgraca();
</script>

</body>
</html>
"""

class ReusableServer(socketserver.TCPServer):
    allow_reuse_address = True

class MLHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(HTML_KOD.encode('utf-8'))

    def do_POST(self):
        length = int(self.headers.get('content-length', 0))
        body = self.rfile.read(length)
        req = json.loads(body.decode('utf-8'))

        if self.path == '/check_defender':
            p_key = req.get('player_key', '')
            d_name_raw = req.get('defender_name', '')
            d_kljuc = normalizuj_unos_defanzivca(d_name_raw)

            sub = df[(df['player_name'] == p_key) & (df['CLOSEST_DEFENDER'].astype(str).str.lower() == d_kljuc)]
            if len(sub) > 0:
                made = int(sub['FGM'].sum())
                total = int(len(sub))
                pct = round((made / total) * 100, 1)
                dist = round(float(sub['CLOSE_DEF_DIST'].mean()), 1)
                res = {'pronadjen': True, 'ime': formatiraj_za_prikaz(d_kljuc), 'made': made, 'total': total, 'pct': pct, 'dist': dist}
            else:
                res = {'pronadjen': False}

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(res).encode('utf-8'))

        elif self.path == '/predict':
            shot_dist = float(req['dist'])
            pts_type = int(req['pts'])

            df_input = pd.DataFrame([{
                'SHOT_DIST': shot_dist,
                'CLOSE_DEF_DIST': float(req['def_dist']),
                'SHOT_CLOCK': float(req['clock']),
                'DRIBBLES': int(req['dribbles']),
                'TOUCH_TIME': float(req['touch']),
                'PTS_TYPE': pts_type,
                'Is_Home': int(req['is_home']),
                'Height_Inches': float(req['height']),
                'Experience_Num': float(req['exp'])
            }])

            prob = float(ml_model.predict_proba(df_input)[0][1])
            ishod = "POGODAK" if prob >= 0.50 else "PROMAŠAJ"
            xp = round(prob * pts_type, 2)
            def_ime = req.get('defender', 'defanzivca')

            if prob >= 0.50:
                komentar = f"Povoljna pozicija! Šut za {pts_type} poena kada ga čuva {def_ime} ima verovatnoću od {prob*100:.1f}%. Model predviđa POGODAK."
            else:
                komentar = f"Težak šut pod pritiskom ({def_ime} na {req['def_dist']} ft). Model predviđa PROMAŠAJ (verovatnoća {prob*100:.1f}%)."

            res = {
                'prob': f"{prob * 100:.1f}",
                'ishod': ishod,
                'xp': f"{xp:.2f}",
                'komentar': komentar
            }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(res).encode('utf-8'))

PORT = 5050
print("==================================================")
print("🚀 POKREĆEM NBA STUDIO: ŠUTER + NJEGOV DEFANZIVAC...")
print("==================================================")
print(f"\n👉 KLIKNI NA LINK: http://127.0.0.1:{PORT}\n")

try:
    subprocess.Popen(['xdg-open', f'http://127.0.0.1:{PORT}'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except:
    webbrowser.open(f'http://127.0.0.1:{PORT}')

server = ReusableServer(('127.0.0.1', PORT), MLHandler)
try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\nZaustavljeno.")