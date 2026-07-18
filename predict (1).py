import numpy as np
import warnings
from datetime import datetime
import time
import random
import requests
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings('ignore')

print("🏀 KnoxDL NBA Player Props Predictor v4 — Form + H2H")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.espn.com/nba/'
}

GAMES_BACK = 15
H2H_BACK   = 10

_last_req = [time.time()]

def rate_limit(mn=0.8, mx=1.8):
    elapsed = time.time() - _last_req[0]
    if elapsed < mn:
        time.sleep(random.uniform(mn, mx))
    _last_req[0] = time.time()

def create_session():
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=1.5, status_forcelist=[429,500,502,503,504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    return s

SESSION = create_session()

def espn_get(url, params=None):
    rate_limit()
    try:
        r = SESSION.get(url, headers=HEADERS, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except:
        return None

# ─────────────────────────────────────────────────────────────
# 1. TODAY'S GAMES
# ─────────────────────────────────────────────────────────────

def get_todays_games():
    print("\n📅 Fetching today's games...")
    data = espn_get(
        "https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
        params={'dates': datetime.now().strftime('%Y%m%d'), 'region': 'us', 'lang': 'en'}
    )
    if not data:
        return []

    games = []
    for event in data.get('events', []):
        comp = event.get('competitions', [{}])[0]
        competitors = comp.get('competitors', [])
        if len(competitors) < 2:
            continue
        away_c = competitors[0]
        home_c = competitors[1]
        status = event.get('status', {}).get('type', {}).get('description', 'Scheduled')
        games.append({
            'home':         home_c['team']['abbreviation'],
            'away':         away_c['team']['abbreviation'],
            'home_espn_id': home_c['team']['id'],
            'away_espn_id': away_c['team']['id'],
            'game_id':      event['id'],
            'time':         event.get('date', ''),
            'status':       status
        })

    upcoming = [g for g in games if any(x in g['status'] for x in ['Scheduled','PM','AM','ET','PT'])]
    result = upcoming if upcoming else games
    print(f"✅ {len(result)} game(s) today")
    return result

# ─────────────────────────────────────────────────────────────
# 2. INJURIES
# ─────────────────────────────────────────────────────────────

_injury_cache = None

def get_injured_players():
    global _injury_cache
    if _injury_cache is not None:
        return _injury_cache

    print("\n🏥 Fetching injury report...")
    data = espn_get("https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/injuries")
    out = set()

    if data:
        for team in data.get('injuries', []):
            for injury in team.get('injuries', []):
                status = injury.get('status', '').lower()
                if any(s in status for s in ['out', 'doubtful', 'suspended', 'inactive']):
                    name = injury.get('athlete', {}).get('displayName', '')
                    if name:
                        out.add(name)

    _injury_cache = out
    print(f"✅ {len(out)} players flagged as out/doubtful")
    return out

# ─────────────────────────────────────────────────────────────
# 3. SCHEDULE
# ─────────────────────────────────────────────────────────────

_schedule_cache = {}

def get_team_schedule(team_espn_id, team_abbrev):
    """Returns list of { id, opponent_id } for completed games, most recent first."""
    if team_abbrev in _schedule_cache:
        return _schedule_cache[team_abbrev]

    print(f"  📅 Fetching schedule for {team_abbrev}...")
    data = espn_get(
        f"https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_espn_id}/schedule"
    )
    if not data:
        _schedule_cache[team_abbrev] = []
        return []

    completed = []
    for event in data.get('events', []):
        comp = event.get('competitions', [{}])[0]
        if comp.get('status', {}).get('type', {}).get('name', '') != 'STATUS_FINAL':
            continue
        opponent_id = None
        for c in comp.get('competitors', []):
            if str(c.get('team', {}).get('id', '')) != str(team_espn_id):
                opponent_id = str(c.get('team', {}).get('id', ''))
                break
        completed.append({'id': event['id'], 'opponent_id': opponent_id})

    result = list(reversed(completed))
    _schedule_cache[team_abbrev] = result
    print(f"  ✅ {len(result)} completed games for {team_abbrev}")
    return result

def get_recent_game_ids(team_espn_id, team_abbrev, n=GAMES_BACK):
    return [g['id'] for g in get_team_schedule(team_espn_id, team_abbrev)[:n]]

def get_h2h_game_ids(team_espn_id, team_abbrev, opponent_espn_id, n=H2H_BACK):
    schedule = get_team_schedule(team_espn_id, team_abbrev)
    return [g['id'] for g in schedule if g['opponent_id'] == str(opponent_espn_id)][:n]

# ─────────────────────────────────────────────────────────────
# 4. PARSE BOXSCORE
# ─────────────────────────────────────────────────────────────

_boxscore_cache = {}

def parse_boxscore(game_id, team_espn_id=None):
    if game_id in _boxscore_cache:
        cached = _boxscore_cache[game_id]
        if team_espn_id:
            return cached['players'].get(str(team_espn_id), {}), cached['scores']
        return cached

    data = espn_get(
        "https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/summary",
        params={'event': game_id}
    )
    if not data:
        return ({}, {}) if team_espn_id else {'players': {}, 'scores': {}}

    boxscore = data.get('boxscore', {})

    scores = {}
    for comp in data.get('header', {}).get('competitions', []):
        for competitor in comp.get('competitors', []):
            tid = str(competitor.get('team', {}).get('id', ''))
            try:
                scores[tid] = float(competitor.get('score', 0))
            except:
                pass

    all_players = {}
    for group in boxscore.get('players', []):
        tid = str(group.get('team', {}).get('id', ''))
        all_players[tid] = {}

        for stat_group in group.get('statistics', []):
            labels = stat_group.get('labels', [])
            if not labels:
                continue
            label_map = {lbl: i for i, lbl in enumerate(labels)}

            def get_val(stats, key):
                idx = label_map.get(key)
                if idx is None or idx >= len(stats):
                    return 0.0
                val = stats[idx]
                try:
                    if isinstance(val, str) and '-' in val and not val.startswith('-'):
                        return float(val.split('-')[0])
                    return float(val) if val not in ('', '--', None) else 0.0
                except:
                    return 0.0

            for athlete_data in stat_group.get('athletes', []):
                if athlete_data.get('didNotPlay'):
                    continue
                name    = athlete_data.get('athlete', {}).get('displayName', '')
                pos_obj = athlete_data.get('athlete', {}).get('position', {})
                pos     = pos_obj.get('abbreviation', '') if isinstance(pos_obj, dict) else ''
                stats   = athlete_data.get('stats', [])
                if not stats or not name:
                    continue
                minutes = get_val(stats, 'MIN')
                if minutes < 1:
                    continue

                all_players[tid][name] = {
                    'PTS': get_val(stats, 'PTS'),
                    'REB': get_val(stats, 'REB'),
                    'AST': get_val(stats, 'AST'),
                    'STL': get_val(stats, 'STL'),
                    'BLK': get_val(stats, 'BLK'),
                    'TOV': get_val(stats, 'TO'),
                    'MIN': minutes,
                    'position': pos,
                }

    result = {'players': all_players, 'scores': scores}
    _boxscore_cache[game_id] = result

    if team_espn_id:
        return all_players.get(str(team_espn_id), {}), scores
    return result

# ─────────────────────────────────────────────────────────────
# 5. DEFENSIVE RATINGS
# ─────────────────────────────────────────────────────────────

_def_ratings = {}

def build_defensive_ratings(all_team_ids):
    print("\n🛡️  Building defensive ratings...")
    for team_espn_id, team_abbrev in all_team_ids:
        game_ids = get_recent_game_ids(team_espn_id, team_abbrev)
        pts_allowed = []
        for game_id in game_ids:
            _, scores = parse_boxscore(game_id, team_espn_id)
            for tid, pts in scores.items():
                if str(tid) != str(team_espn_id):
                    pts_allowed.append(pts)
                    break
        if pts_allowed:
            _def_ratings[str(team_espn_id)] = float(np.mean(pts_allowed))

    if _def_ratings:
        league_avg = float(np.mean(list(_def_ratings.values())))
        print(f"✅ Defensive ratings built — league avg: {league_avg:.1f} PPG allowed")
        return league_avg
    return 113.0

def get_defensive_factor(opponent_espn_id, league_avg):
    avg = _def_ratings.get(str(opponent_espn_id))
    if avg is None or league_avg == 0:
        return 1.0
    return float(max(0.85, min(1.15, avg / league_avg)))

# ─────────────────────────────────────────────────────────────
# 6. PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────

def weighted_average(values, decay=0.92):
    n = len(values)
    if n == 0:
        return 0.0
    weights = np.array([decay ** i for i in range(n)])
    weights /= weights.sum()
    return float(np.dot(weights, values))

def make_prediction(logs, opponent_espn_id, league_avg):
    """
    Given a list of game log dicts, produce a prediction dict.
    Returns None if not enough data.
    """
    if not logs or len(logs) < 2:
        return None

    def extr(key):
        return [g[key] for g in logs]

    pts  = extr('PTS')
    reb  = extr('REB')
    ast  = extr('AST')
    stl  = extr('STL')
    blk  = extr('BLK')
    tov  = extr('TOV')
    mins = extr('MIN')
    pos  = logs[0].get('position', '')

    min_arr       = np.array(mins)
    min_mean      = min_arr.mean()
    min_std       = min_arr.std()
    role_unstable = bool((min_std / max(min_mean, 1)) > 0.40)

    def_factor    = get_defensive_factor(opponent_espn_id, league_avg)
    games_factor  = min(len(logs) / 12, 1.0)
    consist       = 1.0 - min(min_std / max(min_mean, 1), 0.5)
    confidence    = int(round((games_factor * 0.5 + consist * 0.5) * 100))

    return {
        'points':         round(weighted_average(pts)  * def_factor, 1),
        'rebounds':       round(weighted_average(reb)  * def_factor, 1),
        'assists':        round(weighted_average(ast)  * def_factor, 1),
        'steals':         round(weighted_average(stl),  1),
        'blocks':         round(weighted_average(blk),  1),
        'turnovers':      round(weighted_average(tov),  1),
        'minutes':        round(weighted_average(mins), 1),
        'games_analyzed': int(len(logs)),
        'confidence':     confidence,
        'role_unstable':  role_unstable,
        'def_factor':     round(def_factor, 3),
        'position':       str(pos),
    }

# ─────────────────────────────────────────────────────────────
# 7. BUILD LOGS + PROCESS TEAM
# ─────────────────────────────────────────────────────────────

def build_logs_from_ids(game_ids, team_espn_id):
    """Fetch boxscores for a list of game IDs and return player logs dict."""
    player_logs = {}
    for game_id in game_ids:
        box, _ = parse_boxscore(game_id, team_espn_id)
        for player, stats in box.items():
            player_logs.setdefault(player, []).append(stats)
    return player_logs

def process_team(team_espn_id, team_abbrev, opponent_espn_id, league_avg, injured_players):
    """
    Returns two dicts: form_preds, h2h_preds
    Each is { player_name: prediction_dict }
    """
    # General recent form
    recent_ids = get_recent_game_ids(team_espn_id, team_abbrev)
    print(f"  📊 {team_abbrev} — recent form: {len(recent_ids)} games")
    general_logs = build_logs_from_ids(recent_ids, team_espn_id)

    # H2H history vs this specific opponent
    h2h_ids = get_h2h_game_ids(team_espn_id, team_abbrev, opponent_espn_id)
    print(f"  🔁 {team_abbrev} — H2H vs opponent: {len(h2h_ids)} games")
    h2h_logs = build_logs_from_ids(h2h_ids, team_espn_id)

    form_preds = {}
    h2h_preds  = {}

    all_players = set(general_logs.keys()) | set(h2h_logs.keys())

    for player_name in all_players:
        if player_name in injured_players:
            continue

        # Form prediction (need at least 3 games)
        gen = general_logs.get(player_name, [])
        if len(gen) >= 3:
            pred = make_prediction(gen, opponent_espn_id, league_avg)
            if pred:
                form_preds[player_name] = pred

        # H2H prediction (need at least 2 games)
        h2h = h2h_logs.get(player_name, [])
        if len(h2h) >= 2:
            pred = make_prediction(h2h, opponent_espn_id, league_avg)
            if pred:
                h2h_preds[player_name] = pred

    # Sort both by points descending
    form_preds = dict(sorted(form_preds.items(), key=lambda x: x[1]['points'], reverse=True))
    h2h_preds  = dict(sorted(h2h_preds.items(),  key=lambda x: x[1]['points'], reverse=True))

    print(f"  ✅ {len(form_preds)} form predictions, {len(h2h_preds)} h2h predictions")
    return form_preds, h2h_preds

# ─────────────────────────────────────────────────────────────
# 8. OVER/UNDER — separate form + h2h totals
# ─────────────────────────────────────────────────────────────

_pts_scored = {}

def build_pts_scored(all_team_ids):
    for team_espn_id, team_abbrev in all_team_ids:
        game_ids = get_recent_game_ids(team_espn_id, team_abbrev)
        pts_list = []
        for game_id in game_ids:
            cached = parse_boxscore(game_id)
            scores = cached.get('scores', {})
            if str(team_espn_id) in scores:
                pts_list.append(scores[str(team_espn_id)])
        if pts_list:
            _pts_scored[str(team_espn_id)] = float(np.mean(pts_list))

def predict_totals(home_espn_id, away_espn_id, home_abbrev, away_abbrev, league_avg):
    """
    Returns two separate totals:
      form_total: based on each team's recent scoring avg + def factor
      h2h_total:  based on actual scores from past matchups between these teams
    """
    # ── Form-based total ─────────────────────────────────────
    away_scored = _pts_scored.get(str(away_espn_id))
    home_scored = _pts_scored.get(str(home_espn_id))

    form_result = None
    if away_scored is not None and home_scored is not None:
        home_def      = get_defensive_factor(home_espn_id, league_avg)
        away_def      = get_defensive_factor(away_espn_id, league_avg)
        away_pts_pred = round(away_scored * home_def, 1)
        home_pts_pred = round(home_scored * away_def, 1)
        form_result = {
            'predicted_total': round(away_pts_pred + home_pts_pred, 1),
            'away_pts_pred':   away_pts_pred,
            'home_pts_pred':   home_pts_pred,
        }

    # ── H2H-based total ──────────────────────────────────────
    h2h_ids = get_h2h_game_ids(home_espn_id, home_abbrev, away_espn_id)
    h2h_result = None

    if h2h_ids:
        game_totals    = []
        home_pts_list  = []
        away_pts_list  = []

        for game_id in h2h_ids:
            cached = parse_boxscore(game_id)
            scores = cached.get('scores', {})
            if len(scores) == 2:
                vals = list(scores.items())
                # Figure out which score belongs to home vs away
                for tid, pts in vals:
                    if str(tid) == str(home_espn_id):
                        home_pts_list.append(pts)
                    else:
                        away_pts_list.append(pts)
                game_totals.append(sum(scores.values()))

        if game_totals:
            h2h_result = {
                'predicted_total': round(float(np.mean(game_totals)), 1),
                'away_pts_pred':   round(float(np.mean(away_pts_list)), 1) if away_pts_list else None,
                'home_pts_pred':   round(float(np.mean(home_pts_list)), 1) if home_pts_list else None,
                'h2h_games_used':  int(len(game_totals)),
            }
            print(f"  📊 H2H total: {h2h_result['predicted_total']} pts avg over {len(game_totals)} games")

    return form_result, h2h_result

# ─────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────

def main():
    games = get_todays_games()

    if not games:
        print("❌ No games today.")
        with open('predictions.json', 'w') as f:
            json.dump({'generated_at': datetime.now().isoformat(), 'season': '2025-26', 'games': []}, f, indent=2)
        return

    injured_players = get_injured_players()

    all_team_ids = []
    seen = set()
    for g in games:
        for tid, abbrev in [(g['home_espn_id'], g['home']), (g['away_espn_id'], g['away'])]:
            if tid not in seen:
                all_team_ids.append((tid, abbrev))
                seen.add(tid)

    league_avg = build_defensive_ratings(all_team_ids)
    build_pts_scored(all_team_ids)

    all_games_output = []

    for i, game in enumerate(games):
        home    = game['home']
        away    = game['away']
        home_id = game['home_espn_id']
        away_id = game['away_espn_id']
        label   = f"{away} @ {home}"

        print(f"\n{'='*60}")
        print(f"🏀 Game {i+1}/{len(games)}: {label}")
        print(f"{'='*60}")

        print(f"\n  👥 {away} (away)...")
        away_form, away_h2h = process_team(away_id, away, home_id, league_avg, injured_players)

        print(f"\n  👥 {home} (home)...")
        home_form, home_h2h = process_team(home_id, home, away_id, league_avg, injured_players)

        form_total, h2h_total = predict_totals(home_id, away_id, home, away, league_avg)

        all_games_output.append({
            'game':             label,
            'home_team':        home,
            'away_team':        away,
            'game_time':        game.get('time', ''),
            # Form-based predictions
            'form_total':       form_total,
            'home_form':        home_form,
            'away_form':        away_form,
            # H2H-based predictions
            'h2h_total':        h2h_total,
            'home_h2h':         home_h2h,
            'away_h2h':         away_h2h,
        })

    output = {
        'generated_at': datetime.now().isoformat(),
        'season':       '2025-26',
        'games':        all_games_output
    }

    with open('predictions.json', 'w') as f:
        json.dump(output, f, indent=2)

    total_form = sum(len(g['home_form']) + len(g['away_form']) for g in all_games_output)
    total_h2h  = sum(len(g['home_h2h'])  + len(g['away_h2h'])  for g in all_games_output)
    print(f"\n✅ Done — {len(all_games_output)} games")
    print(f"   Form predictions: {total_form} players")
    print(f"   H2H  predictions: {total_h2h} players")
    print(f"   → predictions.json")

if __name__ == "__main__":
    main()
