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

print("🏀 KnoxDL NBA Player Props Predictor — ESPN Boxscore Edition v2")
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

GAMES_BACK = 15  # recent games per team for player logs + def ratings

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
# 2. INJURIES — fetch ESPN injury report
# Returns set of player display names who are OUT or DOUBTFUL
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
                # Flag anyone who is out, doubtful, or suspended
                if any(s in status for s in ['out', 'doubtful', 'suspended', 'inactive']):
                    name = injury.get('athlete', {}).get('displayName', '')
                    if name:
                        out.add(name)

    _injury_cache = out
    print(f"✅ {len(out)} players flagged as out/doubtful")
    return out

# ─────────────────────────────────────────────────────────────
# 3. RECENT GAME IDs PER TEAM
# ─────────────────────────────────────────────────────────────

_schedule_cache = {}

def get_recent_game_ids(team_espn_id, team_abbrev, n=GAMES_BACK):
    if team_abbrev in _schedule_cache:
        return _schedule_cache[team_abbrev]

    print(f"  📅 Fetching schedule for {team_abbrev}...")
    data = espn_get(
        f"https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_espn_id}/schedule"
    )
    if not data:
        return []

    completed = []
    for event in data.get('events', []):
        status = event.get('competitions', [{}])[0].get('status', {}).get('type', {}).get('name', '')
        if status == 'STATUS_FINAL':
            completed.append(event['id'])

    recent = list(reversed(completed))[:n]
    print(f"  ✅ {len(recent)} recent games for {team_abbrev}")
    _schedule_cache[team_abbrev] = recent
    return recent

# ─────────────────────────────────────────────────────────────
# 4. PARSE BOXSCORE
# Returns:
#   players: { name: { PTS, REB, AST, STL, BLK, TOV, MIN, position } }
#   team_scores: { team_espn_id: points_scored }
# ─────────────────────────────────────────────────────────────

_boxscore_cache = {}

def parse_boxscore(game_id, team_espn_id=None):
    if game_id in _boxscore_cache:
        cached = _boxscore_cache[game_id]
        if team_espn_id:
            return cached.get('players', {}).get(str(team_espn_id), {}), cached.get('scores', {})
        return cached

    data = espn_get(
        "https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/summary",
        params={'event': game_id}
    )
    if not data:
        return ({}, {}) if team_espn_id else {}

    boxscore = data.get('boxscore', {})

    # ── Extract scores per team ──────────────────────────────
    scores = {}
    for team_data in boxscore.get('teams', []):
        tid   = str(team_data.get('team', {}).get('id', ''))
        score = team_data.get('score', 0)
        try:
            scores[tid] = float(score)
        except:
            pass

    # ── Extract player stats per team ───────────────────────
    all_players = {}  # { team_id_str: { player_name: stats } }

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
                name  = athlete_data.get('athlete', {}).get('displayName', '')
                pos   = athlete_data.get('athlete', {}).get('position', {})
                pos   = pos.get('abbreviation', '') if isinstance(pos, dict) else ''
                stats = athlete_data.get('stats', [])
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

    _boxscore_cache[game_id] = {'players': all_players, 'scores': scores}

    if team_espn_id:
        return all_players.get(str(team_espn_id), {}), scores
    return {'players': all_players, 'scores': scores}

# ─────────────────────────────────────────────────────────────
# 5. OPPONENT DEFENSIVE RATING
# For each team, calculate average points they ALLOW per game
# over their last GAMES_BACK games — using actual boxscore scores.
# Returns { team_espn_id: avg_pts_allowed }
# ─────────────────────────────────────────────────────────────

_def_ratings = {}

def build_defensive_ratings(all_team_ids):
    """
    Pre-fetch all schedules and boxscores to calculate how many
    points each team allows on average. Uses the same boxscore
    cache so we don't double-fetch anything.
    """
    print("\n🛡️  Building opponent defensive ratings...")

    for team_espn_id, team_abbrev in all_team_ids:
        game_ids = get_recent_game_ids(team_espn_id, team_abbrev)
        pts_allowed = []

        for game_id in game_ids:
            _, scores = parse_boxscore(game_id, team_espn_id)
            # Points allowed = points scored by the OTHER team
            for tid, pts in scores.items():
                if str(tid) != str(team_espn_id):
                    pts_allowed.append(pts)
                    break

        if pts_allowed:
            avg = float(np.mean(pts_allowed))
            _def_ratings[str(team_espn_id)] = avg

    # Calculate league average for relative scaling
    if _def_ratings:
        league_avg = float(np.mean(list(_def_ratings.values())))
        print(f"✅ Defensive ratings built — league avg allowed: {league_avg:.1f} PPG")
        return league_avg
    return 113.0  # fallback league average

def get_defensive_factor(opponent_espn_id, league_avg):
    """
    Returns a scalar:
      > 1.0 = opponent allows more than average (easier matchup)
      < 1.0 = opponent is stingy (harder matchup)
    Capped at ±15% to avoid overreaction.
    """
    avg_allowed = _def_ratings.get(str(opponent_espn_id))
    if avg_allowed is None or league_avg == 0:
        return 1.0
    factor = avg_allowed / league_avg
    return float(max(0.85, min(1.15, factor)))

# ─────────────────────────────────────────────────────────────
# 6. BUILD PLAYER LOGS
# ─────────────────────────────────────────────────────────────

def build_player_logs(team_espn_id, team_abbrev):
    game_ids = get_recent_game_ids(team_espn_id, team_abbrev)
    if not game_ids:
        return {}

    player_logs = {}
    print(f"  📊 Building player logs for {team_abbrev} from {len(game_ids)} games...")

    for game_id in game_ids:
        box, _ = parse_boxscore(game_id, team_espn_id)
        for player, stats in box.items():
            player_logs.setdefault(player, []).append(stats)

    print(f"  ✅ {len(player_logs)} players tracked for {team_abbrev}")
    return player_logs

# ─────────────────────────────────────────────────────────────
# 7. PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────

def weighted_average(values, decay=0.92):
    n = len(values)
    if n == 0:
        return 0.0
    weights = np.array([decay ** i for i in range(n)])
    weights /= weights.sum()
    return float(np.dot(weights, values))

def predict_player(logs, player_name, opponent_espn_id, league_avg, injured_players):
    if not logs or len(logs) < 3:
        return None

    # ── Injury check ────────────────────────────────────────
    is_injured = player_name in injured_players
    if is_injured:
        return None  # exclude injured/doubtful players entirely

    pts  = [g['PTS'] for g in logs]
    reb  = [g['REB'] for g in logs]
    ast  = [g['AST'] for g in logs]
    stl  = [g['STL'] for g in logs]
    blk  = [g['BLK'] for g in logs]
    tov  = [g['TOV'] for g in logs]
    mins = [g['MIN'] for g in logs]
    pos  = logs[0].get('position', '')

    min_arr       = np.array(mins)
    min_mean      = min_arr.mean()
    min_std       = min_arr.std()
    role_unstable = bool((min_std / max(min_mean, 1)) > 0.40)

    # ── Base weighted averages ───────────────────────────────
    pts_base = weighted_average(pts)
    reb_base = weighted_average(reb)
    ast_base = weighted_average(ast)
    stl_base = weighted_average(stl)
    blk_base = weighted_average(blk)
    tov_base = weighted_average(tov)
    min_base = weighted_average(mins)

    # ── Opponent defensive factor ────────────────────────────
    def_factor = get_defensive_factor(opponent_espn_id, league_avg)

    # Apply def_factor only to offensive stats (pts, reb, ast)
    # STL/BLK/TOV are less affected by opponent offense quality
    pts_pred = round(pts_base * def_factor, 1)
    reb_pred = round(reb_base * def_factor, 1)
    ast_pred = round(ast_base * def_factor, 1)
    stl_pred = round(stl_base, 1)
    blk_pred = round(blk_base, 1)
    tov_pred = round(tov_base, 1)
    min_pred = round(min_base, 1)

    # ── Confidence ───────────────────────────────────────────
    games_factor   = min(len(logs) / 12, 1.0)
    consist_factor = 1.0 - min(min_std / max(min_mean, 1), 0.5)
    confidence     = int(round((games_factor * 0.5 + consist_factor * 0.5) * 100))

    return {
        'points':         pts_pred,
        'rebounds':       reb_pred,
        'assists':        ast_pred,
        'steals':         stl_pred,
        'blocks':         blk_pred,
        'turnovers':      tov_pred,
        'minutes':        min_pred,
        'games_analyzed': int(len(logs)),
        'confidence':     confidence,
        'role_unstable':  role_unstable,
        'def_factor':     round(def_factor, 3),
        'position':       str(pos),
    }


# ─────────────────────────────────────────────────────────────
# 8. OVER/UNDER TOTAL PREDICTION
# Uses each team's avg pts scored + opponent's defensive factor
# ─────────────────────────────────────────────────────────────

_pts_scored = {}

def build_pts_scored(all_team_ids):
    """Calculate average points scored per team from their recent boxscores."""
    for team_espn_id, team_abbrev in all_team_ids:
        game_ids = get_recent_game_ids(team_espn_id, team_abbrev)
        pts_list = []
        for game_id in game_ids:
            _, scores = parse_boxscore(game_id, team_espn_id)
            if str(team_espn_id) in scores:
                pts_list.append(scores[str(team_espn_id)])
        if pts_list:
            _pts_scored[str(team_espn_id)] = float(np.mean(pts_list))

def predict_over_under(home_espn_id, away_espn_id, league_avg):
    """
    Predicted total = away avg pts scored × home def factor
                    + home avg pts scored × away def factor
    """
    away_scored = _pts_scored.get(str(away_espn_id))
    home_scored = _pts_scored.get(str(home_espn_id))
    if away_scored is None or home_scored is None:
        return None

    home_def      = get_defensive_factor(home_espn_id, league_avg)
    away_def      = get_defensive_factor(away_espn_id, league_avg)
    away_pts_pred = round(away_scored * home_def, 1)
    home_pts_pred = round(home_scored * away_def, 1)
    total         = round(away_pts_pred + home_pts_pred, 1)

    return {
        'predicted_total': total,
        'away_pts_pred':   away_pts_pred,
        'home_pts_pred':   home_pts_pred,
    }

# ─────────────────────────────────────────────────────────────
# 8. PROCESS A TEAM
# ─────────────────────────────────────────────────────────────

def process_team(team_espn_id, team_abbrev, opponent_espn_id, league_avg, injured_players):
    player_logs = build_player_logs(team_espn_id, team_abbrev)
    results = {}

    for player_name, logs in player_logs.items():
        pred = predict_player(logs, player_name, opponent_espn_id, league_avg, injured_players)
        if pred:
            results[player_name] = pred
            flag = " ⚠️ unstable" if pred['role_unstable'] else ""
            df   = f" [def×{pred['def_factor']}]" if pred['def_factor'] != 1.0 else ""
            print(f"    ✅ {player_name}: {pred['points']} PTS | {pred['rebounds']} REB | {pred['assists']} AST [{pred['confidence']}% conf]{flag}{df}")

    return dict(sorted(results.items(), key=lambda x: x[1]['points'], reverse=True))

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

    # Fetch injuries once upfront
    injured_players = get_injured_players()

    # Collect all unique teams playing today
    all_team_ids = []
    seen = set()
    for g in games:
        for tid, abbrev in [(g['home_espn_id'], g['home']), (g['away_espn_id'], g['away'])]:
            if tid not in seen:
                all_team_ids.append((tid, abbrev))
                seen.add(tid)

    # Build defensive ratings for all teams (reuses boxscore cache)
    league_avg = build_defensive_ratings(all_team_ids)

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

        print(f"\n  👥 {away} (away) vs {home} defense...")
        away_preds = process_team(away_id, away, home_id, league_avg, injured_players)

        print(f"\n  👥 {home} (home) vs {away} defense...")
        home_preds = process_team(home_id, home, away_id, league_avg, injured_players)

        all_games_output.append({
            'game':         label,
            'home_team':    home,
            'away_team':    away,
            'game_time':    game.get('time', ''),
            'home_players': home_preds,
            'away_players': away_preds,
        })

    output = {
        'generated_at': datetime.now().isoformat(),
        'season':       '2025-26',
        'games':        all_games_output
    }

    with open('predictions.json', 'w') as f:
        json.dump(output, f, indent=2)

    total = sum(len(g['home_players']) + len(g['away_players']) for g in all_games_output)
    print(f"\n✅ Done — {len(all_games_output)} games, {total} players → predictions.json")

if __name__ == "__main__":
    main()
