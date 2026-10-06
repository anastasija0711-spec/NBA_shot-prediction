import os
import sys
import pandas as pd

trenutni_folder = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(trenutni_folder) == 'models':
    glavni_folder = os.path.dirname(trenutni_folder)
else:
    glavni_folder = trenutni_folder

putanja_sutevi = os.path.join(glavni_folder, 'data', 'shot_logs.csv')
putanja_igraci = os.path.join(glavni_folder, 'data', 'players.csv')

print("--- PROVERA FAJLOVA ---")
for naziv, putanja in [('shot_logs.csv', putanja_sutevi), ('players.csv', putanja_igraci)]:
    if not os.path.exists(putanja):
        print(f"GREŠKA: Fajl '{naziv}' ne postoji na putanji: {putanja}")
        sys.exit(1)
    
    velicina = os.path.getsize(putanja)
    print(f"✔️ Fajl: {naziv} | Veličina: {velicina / 1024:.1f} KB")
    
    if velicina == 0:
        print(f"\n GREŠKA: Fajl '{naziv}' je PRAZAN (ima 0 bajtova)!")
        print("Rešenje: Verovatno si slučajno obrisao sadržaj dok ti je bio otvoren u tabu.")
        print("1. Pogledaj gore u tabove da li ti je otvoren 'shot_logs.csv'.")
        print("2. Ako jeste, zatvori ga na X i klikni 'Don't Save' ili stisni Ctrl + Z da vratiš podatke.")
        print("3. Ili ponovo prevuci originalni fajl iz Downloads foldera u folder 'data/'.\n")
        sys.exit(1)


# Učitavanje podataka
print("Učitavanje podataka...")
shots_df = pd.read_csv(putanja_sutevi)
players_df = pd.read_csv(putanja_igraci)

# Formatiranje imena
shots_df['player_name'] = shots_df['player_name'].astype(str).str.lower().str.strip()
players_df['Name'] = players_df['Name'].astype(str).str.lower().str.strip()

# Uklanjanje duplikata igrača
players_clean = players_df.drop_duplicates(subset=['Name'], keep='last')

# Izdvajanje potrebnih kolona
zeljene_kolone = [k for k in ['Name', 'Pos', 'Height', 'Experience'] if k in players_clean.columns]
players_clean = players_clean[zeljene_kolone]


df = pd.merge(shots_df, players_clean, left_on='player_name', right_on='Name', how='left')

if 'Pos' in df.columns:
    df['Pos'] = df['Pos'].fillna('N/A')
if 'Height' in df.columns:
    df['Height'] = df['Height'].fillna('N/A')
if 'Experience' in df.columns:
    df['Experience'] = df['Experience'].fillna('-')

df['FGM'] = pd.to_numeric(df['FGM'], errors='coerce').fillna(0).astype(int)


def normalizuj_defanzivca(unos):
    unos = unos.strip().lower()
    if not unos:
        return ""
    if ',' in unos:
        delovi = [p.strip() for p in unos.split(',')]
        return f"{delovi[0]}, {delovi[1]}" if len(delovi) >= 2 else unos
    delovi = unos.split()
    if len(delovi) >= 2:
        return f"{delovi[-1]}, {' '.join(delovi[:-1])}"
    return unos

def formatiraj_ime_za_prikaz(ime_baze):
    if ',' in ime_baze:
        prezime, ime = [p.strip() for p in ime_baze.split(',')]
        return f"{ime.title()} {prezime.title()}"
    return ime_baze.title()

def analiziraj_uspesnost_napredno(podaci):
    print("\n=== NAPREDNA ANALIZA ŠUTA ===")
    igrac = input("Unesi ime igrača (npr. lebron james): ").strip().lower()
    
    tip_suta_unos = input("Unesi tip šuta (2 za dvojku, 3 za trojku): ").strip()
    try:
        tip_suta = int(tip_suta_unos)
        if tip_suta not in [2, 3]:
            print("Greška: Dozvoljeni su samo brojevi 2 ili 3.")
            return
    except ValueError:
        print("Greška: Moraš uneti broj 2 ili 3.")
        return

    defanzivac_raw = input("Unesi ime defanzivca (npr. 'Jimmy Butler' ili ostavi prazno): ").strip()
    defanzivac_kljuc = normalizuj_defanzivca(defanzivac_raw)

    # Filtriranje
    filter_podaci = podaci[
        (podaci['player_name'] == igrac) & 
        (podaci['PTS_TYPE'] == tip_suta)
    ]

    if defanzivac_kljuc != "":
        filter_podaci = filter_podaci[filter_podaci['CLOSEST_DEFENDER'].astype(str).str.lower() == defanzivac_kljuc]
        prikaz_defanzivca = formatiraj_ime_za_prikaz(defanzivac_kljuc)
        poruka_odbrana = f"kada ga čuva {prikaz_defanzivca}"
    else:
        prikaz_defanzivca = "Svi"
        poruka_odbrana = "protiv svih defanzivaca"

    ukupno_suteva = len(filter_podaci)
    if ukupno_suteva == 0:
        print(f"\nNema podataka za igrača {igrac.title()} za {tip_suta} poena {poruka_odbrana}.")
        return

    pogodci = filter_podaci['FGM'].sum()
    uspesnost = (pogodci / ukupno_suteva) * 100

    pozicija = filter_podaci['Pos'].iloc[0] if 'Pos' in filter_podaci.columns else 'N/A'
    visina = filter_podaci['Height'].iloc[0] if 'Height' in filter_podaci.columns else 'N/A'
    iskustvo = filter_podaci['Experience'].iloc[0] if 'Experience' in filter_podaci.columns else '-'

    print("\n--- REZULTATI ---")
    print(f"Igrač: {igrac.title()} | Pozicija: {pozicija} | Visina: {visina} | Iskustvo: {iskustvo} god.")
    print(f"Tip šuta: Za {tip_suta} poena")
    print(f"Defanzivac: {prikaz_defanzivca}")
    print(f"Šutirao: {ukupno_suteva} puta")
    print(f"Pogodio: {int(pogodci)} puta")
    print(f"Uspešnost: {uspesnost:.2f}%")
    print("-----------------\n")

while True:
    analiziraj_uspesnost_napredno(df)
    jos = input("Da li želiš da proveriš još neku kombinaciju? (da/ne): ").strip().lower()
    if jos != 'da':
        print("Kraj programa.")
        break