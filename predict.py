import numpy as np
import warnings
from datetime import datetime
import time
import random
import requests
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from zoneinfo import ZoneInfo
    NBA_TZ = ZoneInfo('America/New_York')
except Exception:
    NBA_TZ = None

warnings.filterwarnings('ignore')

print("🏀 KnoxDL NBA Player Props Predictor v6 — Form + H2H + Rest")
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

BASE = "https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba"

GAMES_BACK        = 15   # recent-form window (current season only)
H2H_BACK          = 10   # max head-to-head games used
H2H_SEASONS_BACK  = 1    # how many PREVIOUS seasons to search for H2H games
MIN_FORM_GAMES    = 3
MIN_H2H_GAMES     = 2
H2H_BLEND_K       = 6    # 'best' = form blended with H2H, weight n/(n+K)
DEF_SHRINK_K      = 10   # shrink defensive factor toward 1.0 on small samples

# Rest / fatigue adjustments (multipliers on a team's own production)
B2B_FACTOR          = 0.97    # second night of a back-to-back
THREE_IN_FOUR_FACTOR= 0.985   # third game in four nights (not a b2b)
RESTED_FACTOR       = 1.01    # two or more full days off
TIRED_OPP_SHARE     = 0.5     # how much of the opponent's fatigue turns into extra points for us


def nba_now():
    """NBA 'today' is the US Eastern date, not the local one (matters in Europe)."""
    return datetime.now(NBA_TZ) if NBA_TZ else datetime.now()


def current_season_year():
    """ESPN names a season by the year it ends in: 2026-27 -> 2027."""
    now = nba_now()
    return now.year + 1 if now.month >= 10 else now.year


SEASON_YEAR  = current_season_year()
SEASON_LABEL = f"{SEASON_YEAR - 1}-{str(SEASON_YEAR)[2:]}"

_last_req = [0.0]
_failed_requests = [0]


def rate_limit(mn=0.5, mx=1.1):
    elapsed = time.time() - _last_req[0]
    if elapsed < mn:
        time.sleep(random.uniform(mn, mx))
    _last_req[0] = time.time()


def create_session():
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504])
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
    except Exception as e:
        # Do not fail silently: a hidden error looks exactly like "no data".
        _failed_requests[0] += 1
        print(f"  ⚠️  Request failed: {url.replace(BASE, '')} {params or ''} — {type(e).__name__}: {e}")
        return None


def to_score(raw):
    """Scoreboard gives scores as '112', the schedule endpoint as {'value': 112.0}."""
    try:
        if isinstance(raw, dict):
            raw = raw.get('value', raw.get('displayValue'))
        if raw in (None, ''):
            return None
        return float(raw)
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────
# 1. TODAY'S GAMES
# ─────────────────────────────────────────────────────────────

def get_todays_games():
    print("\n📅 Fetching today's games...")
    data = espn_get(
        f"{BASE}/scoreboard",
        params={'dates': nba_now().strftime('%Y%m%d'), 'region': 'us', 'lang': 'en'}
    )
    if not data:
        return []

    games = []
    for event in data.get('events', []):
        comp = (event.get('competitions') or [{}])[0]
        competitors = comp.get('competitors', [])
        if len(competitors) < 2:
            continue

        # Use ESPN's homeAway flag instead of trusting list order.
        home_c = next((c for c in competitors if c.get('homeAway') == 'home'), None)
        away_c = next((c for c in competitors if c.get('homeAway') == 'away'), None)
        if home_c is None or away_c is None:
            home_c, away_c = competitors[0], competitors[1]

        stype = event.get('status', {}).get('type', {})
        games.append({
            'home':         home_c['team']['abbreviation'],
            'away':         away_c['team']['abbreviation'],
            'home_espn_id': str(home_c['team']['id']),
            'away_espn_id': str(away_c['team']['id']),
            'game_id':      event['id'],
            'time':         event.get('date', ''),
            'status':       stype.get('description', 'Scheduled'),
            'state':        stype.get('state', ''),
        })

    upcoming = [g for g in games
                if g['state'] == 'pre'
                or any(x in g['status'] for x in ['Scheduled', 'PM', 'AM', 'ET', 'PT'])]
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
    data = espn_get(f"{BASE}/injuries")
    out = set()

    if data:
        for team in data.get('injuries', []):
            for injury in team.get('injuries', []):
                status = str(injury.get('status', '')).lower()
                if any(s in status for s in ['out', 'doubtful', 'suspended', 'inactive']):
                    name = injury.get('athlete', {}).get('displayName', '')
                    if name:
                        out.add(name)

    _injury_cache = out
    print(f"✅ {len(out)} players flagged as out/doubtful")
    return out

# ─────────────────────────────────────────────────────────────
# 3. SCHEDULE  (current season for form, + previous season(s) for H2H)
# ─────────────────────────────────────────────────────────────

_season_sched_cache = {}   # (team_id, season, seasontype) -> list of games


def _fetch_season_schedule(team_espn_id, season, seasontype):
    """Completed games for one team / season / type (2 = regular, 3 = playoffs)."""
    key = (str(team_espn_id), season, seasontype)
    if key in _season_sched_cache:
        return _season_sched_cache[key]

    data = espn_get(
        f"{BASE}/teams/{team_espn_id}/schedule",
        params={'season': season, 'seasontype': seasontype}
    )
    completed = []
    for event in (data or {}).get('events', []):
        comp = (event.get('competitions') or [{}])[0]
        stype = comp.get('status', {}).get('type', {})
        if stype.get('name', '') != 'STATUS_FINAL' and not stype.get('completed', False):
            continue
        opponent_id, pts_for, pts_against = None, None, None
        for c in comp.get('competitors', []):
            tid = str(c.get('team', {}).get('id', ''))
            if tid == str(team_espn_id):
                pts_for = to_score(c.get('score'))
            else:
                opponent_id = tid
                pts_against = to_score(c.get('score'))
        completed.append({
            'id':          str(event['id']),
            'date':        event.get('date', ''),
            'opponent_id': opponent_id,
            'pts_for':     pts_for,
            'pts_against': pts_against,
            'season':      season,
        })

    _season_sched_cache[key] = completed
    return completed


def _dedupe_sorted(games):
    seen, out = set(), []
    for g in sorted(games, key=lambda g: g['date'], reverse=True):
        if g['id'] not in seen:
            seen.add(g['id'])
            out.append(g)
    return out


_schedule_cache = {}
_history_cache = {}


def get_team_schedule(team_espn_id, team_abbrev, quiet=False):
    """Current-season completed games, most recent first."""
    key = str(team_espn_id)
    if key in _schedule_cache:
        return _schedule_cache[key]
    if not quiet:
        print(f"  📅 Fetching {SEASON_LABEL} schedule for {team_abbrev}...")
    games = _fetch_season_schedule(team_espn_id, SEASON_YEAR, 2)
    # Playoff games only exist from mid-April on; skip the request before that.
    if nba_now().month in (4, 5, 6):
        games = games + _fetch_season_schedule(team_espn_id, SEASON_YEAR, 3)
    result = _dedupe_sorted(games)
    _schedule_cache[key] = result
    if not quiet:
        print(f"  ✅ {len(result)} completed games for {team_abbrev}")
    return result


def get_team_history(team_espn_id, team_abbrev):
    """Current season + previous season(s) incl. playoffs, most recent first."""
    key = str(team_espn_id)
    if key in _history_cache:
        return _history_cache[key]
    games = list(get_team_schedule(team_espn_id, team_abbrev))
    for back in range(1, H2H_SEASONS_BACK + 1):
        yr = SEASON_YEAR - back
        print(f"  📅 Fetching {yr - 1}-{str(yr)[2:]} schedule for {team_abbrev} (H2H history)...")
        games += _fetch_season_schedule(team_espn_id, yr, 2)
        games += _fetch_season_schedule(team_espn_id, yr, 3)
    result = _dedupe_sorted(games)
    _history_cache[key] = result
    return result


def get_recent_game_ids(team_espn_id, team_abbrev, n=GAMES_BACK):
    return [g['id'] for g in get_team_schedule(team_espn_id, team_abbrev)[:n]]


def get_h2h_game_ids(team_espn_id, team_abbrev, opponent_espn_id, n=H2H_BACK):
    history = get_team_history(team_espn_id, team_abbrev)
    return [g['id'] for g in history if g['opponent_id'] == str(opponent_espn_id)][:n]

# ─────────────────────────────────────────────────────────────
# 3b. REST DAYS / BACK-TO-BACKS  (from the same schedule feed)
# ─────────────────────────────────────────────────────────────

def _nba_date(iso):
    """ESPN ISO timestamp (UTC) -> calendar date in US Eastern."""
    try:
        dt = datetime.fromisoformat(str(iso).replace('Z', '+00:00'))
        if dt.tzinfo is not None and NBA_TZ:
            dt = dt.astimezone(NBA_TZ)
        return dt.date()
    except Exception:
        return None


def get_rest_info(team_espn_id, team_abbrev, game_time):
    """
    rest_days: full days off before tonight (0 = played yesterday = back-to-back).
    Returns a neutral result if dates cannot be read.
    """
    info = {'rest_days': None, 'back_to_back': False, 'three_in_four': False, 'factor': 1.0}
    today = _nba_date(game_time) or nba_now().date()
    gaps = []
    for g in get_team_schedule(team_espn_id, team_abbrev):
        d = _nba_date(g['date'])
        if d is None:
            continue
        gap = (today - d).days
        if gap >= 1:
            gaps.append(gap)
    if not gaps:
        return info

    last_gap = min(gaps)
    info['rest_days']     = int(last_gap - 1)
    info['back_to_back']  = last_gap == 1
    info['three_in_four'] = sum(1 for x in gaps if x <= 3) >= 2

    if info['back_to_back']:
        info['factor'] = B2B_FACTOR
    elif info['three_in_four']:
        info['factor'] = THREE_IN_FOUR_FACTOR
    elif 2 <= info['rest_days'] <= 6:      # longer gaps = breaks/rust, treat as neutral
        info['factor'] = RESTED_FACTOR
    return info


def tired_opponent_boost(opp_rest):
    """A fatigued opponent defends a little worse."""
    f = opp_rest.get('factor', 1.0)
    return 1.0 + (1.0 - f) * TIRED_OPP_SHARE if f < 1.0 else 1.0

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

    data = espn_get(f"{BASE}/summary", params={'event': game_id})
    if not data:
        return ({}, {}) if team_espn_id else {'players': {}, 'scores': {}}

    boxscore = data.get('boxscore', {})

    scores = {}
    for comp in data.get('header', {}).get('competitions', []):
        for competitor in comp.get('competitors', []):
            tid = str(competitor.get('team', {}).get('id', ''))
            sc = to_score(competitor.get('score'))
            if sc is not None:
                scores[tid] = sc

    all_players = {}
    for group in boxscore.get('players', []):
        tid = str(group.get('team', {}).get('id', ''))
        all_players[tid] = {}

        for stat_group in group.get('statistics', []):
            labels = stat_group.get('labels') or stat_group.get('names') or []
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
                except Exception:
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
# 5. TEAM RATINGS  (all 30 teams, straight from schedule scores)
# ─────────────────────────────────────────────────────────────

_def_ratings = {}    # team_id -> (avg pts allowed, games)
_pts_scored  = {}    # team_id -> avg pts scored


def get_all_teams():
    data = espn_get(f"{BASE}/teams", params={'limit': 50})
    teams = []
    try:
        for t in data['sports'][0]['leagues'][0]['teams']:
            teams.append((str(t['team']['id']), t['team']['abbreviation']))
    except Exception:
        pass
    return teams


def build_team_ratings(todays_team_ids):
    """
    Points scored / allowed over each team's last GAMES_BACK games.
    Uses all 30 teams so 'league average' really is the league average,
    not just the average of whoever happens to play tonight.
    Scores come from the schedule feed, so this costs one request per team.
    """
    print("\n🛡️  Building team ratings (all teams)...")
    teams = get_all_teams()
    known = {tid for tid, _ in teams}
    for tid, ab in todays_team_ids:
        if tid not in known:
            teams.append((tid, ab))

    todays = {tid for tid, _ in todays_team_ids}
    for tid, ab in teams:
        recent = get_team_schedule(tid, ab, quiet=True)[:GAMES_BACK]
        allowed = [g['pts_against'] for g in recent if g['pts_against'] is not None]
        scored  = [g['pts_for']     for g in recent if g['pts_for']     is not None]

        # Fallback to boxscores (today's teams only) if the schedule had no scores.
        if not allowed and tid in todays:
            for g in recent:
                scores = parse_boxscore(g['id']).get('scores', {})
                for stid, pts in scores.items():
                    if stid == tid:
                        scored.append(pts)
                    else:
                        allowed.append(pts)

        if allowed:
            _def_ratings[tid] = (float(np.mean(allowed)), len(allowed))
        if scored:
            _pts_scored[tid] = float(np.mean(scored))

    if _def_ratings:
        league_avg = float(np.mean([v[0] for v in _def_ratings.values()]))
        print(f"✅ Ratings for {len(_def_ratings)} teams — league avg: {league_avg:.1f} PPG allowed")
        return league_avg
    return 113.0


def get_defensive_factor(opponent_espn_id, league_avg):
    entry = _def_ratings.get(str(opponent_espn_id))
    if entry is None or not league_avg:
        return 1.0
    avg, n = entry
    raw = avg / league_avg
    # Early in the season a few games say little: pull the factor toward 1.0.
    shrunk = 1.0 + (raw - 1.0) * (n / (n + DEF_SHRINK_K))
    return float(max(0.85, min(1.15, shrunk)))

# ─────────────────────────────────────────────────────────────
# 6. PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────

STAT_KEYS = [('points', 'PTS'), ('rebounds', 'REB'), ('assists', 'AST'),
             ('steals', 'STL'), ('blocks', 'BLK'), ('turnovers', 'TOV')]


def _weights(n, decay):
    w = np.array([decay ** i for i in range(n)], dtype=float)
    return w / w.sum()


def weighted_average(values, decay=0.92):
    n = len(values)
    if n == 0:
        return 0.0
    return float(np.dot(_weights(n, decay), values))


def make_prediction(logs, def_factor=1.0, full_sample=12, rest_factor=1.0, pts_boost=1.0):
    """
    logs: game dicts, most recent first.

    Projection = expected minutes × per-minute production.
      - Minutes follow the player's current role, so they react fast.
      - Per-minute rates are steadier, so they use a longer memory.
    def_factor: opponent adjustment. Full effect on points, half on assists,
    none on rebounds/steals/blocks (points allowed says nothing about those).
    Pass 1.0 for H2H logs — they are already against this opponent.
    rest_factor: this team's fatigue/rest multiplier (points, rebounds, assists).
    pts_boost:   extra points multiplier when the opponent is fatigued.
    """
    if not logs or len(logs) < 2:
        return None

    n    = len(logs)
    mins = np.array([g['MIN'] for g in logs], dtype=float)
    pos  = logs[0].get('position', '')

    exp_min = 0.5 * weighted_average(mins, 0.80) + 0.5 * weighted_average(mins, 0.95)

    w_rate = _weights(n, 0.95)
    min_w  = float(np.dot(w_rate, mins))

    def project(key):
        vals = np.array([g[key] for g in logs], dtype=float)
        if min_w <= 0:
            return 0.0
        return float(np.dot(w_rate, vals)) / min_w * exp_min

    min_mean      = mins.mean()
    min_std       = mins.std()
    cv            = min_std / max(min_mean, 1)
    role_unstable = bool(cv > 0.40)

    games_factor  = min(n / full_sample, 1.0)
    consist       = 1.0 - min(cv, 0.5)
    confidence    = int(round((games_factor * 0.5 + consist * 0.5) * 100))

    ast_factor = 1.0 + (def_factor - 1.0) * 0.5

    return {
        'points':         round(project('PTS') * def_factor * rest_factor * pts_boost, 1),
        'rebounds':       round(project('REB') * rest_factor, 1),
        'assists':        round(project('AST') * ast_factor * rest_factor, 1),
        'steals':         round(project('STL'), 1),
        'blocks':         round(project('BLK'), 1),
        'turnovers':      round(project('TOV'), 1),
        'minutes':        round(exp_min, 1),
        'games_analyzed': int(n),
        'confidence':     confidence,
        'role_unstable':  role_unstable,
        'def_factor':     round(def_factor, 3),
        'rest_factor':    round(rest_factor * pts_boost, 3),
        'position':       str(pos),
    }


def blend_predictions(form, h2h, rest_factor=1.0, pts_boost=1.0):
    """Form is the base; H2H nudges it in proportion to how many H2H games exist.
    The H2H side is raw history, so tonight's rest adjustment is applied to it here."""
    if not h2h:
        return dict(form, h2h_weight=0.0)
    n = h2h['games_analyzed']
    w = n / (n + H2H_BLEND_K)
    out = dict(form)
    for key in ['points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers']:
        adj = 1.0
        if key in ('points', 'rebounds', 'assists'):
            adj = rest_factor * (pts_boost if key == 'points' else 1.0)
        out[key] = round((1 - w) * form[key] + w * h2h[key] * adj, 1)
    out['h2h_weight'] = round(w, 2)
    return out

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


def process_team(team_espn_id, team_abbrev, opponent_espn_id, league_avg, injured_players,
                 rest_factor=1.0, pts_boost=1.0):
    """Returns three dicts { player_name: prediction }: form, h2h, best (blend)."""
    recent_ids = get_recent_game_ids(team_espn_id, team_abbrev)
    print(f"  📊 {team_abbrev} — recent form: {len(recent_ids)} games")
    general_logs = build_logs_from_ids(recent_ids, team_espn_id)

    h2h_ids = get_h2h_game_ids(team_espn_id, team_abbrev, opponent_espn_id)
    print(f"  🔁 {team_abbrev} — H2H vs opponent: {len(h2h_ids)} games")
    h2h_logs = build_logs_from_ids(h2h_ids, team_espn_id)

    def_factor = get_defensive_factor(opponent_espn_id, league_avg)

    # Only players who have appeared for this team THIS season are on the roster
    # as far as we can tell; that keeps departed players out of the H2H list.
    current_roster = set(general_logs.keys())
    if not current_roster:           # first game of the season: nothing to filter on
        current_roster = set(h2h_logs.keys())

    form_preds, h2h_preds, best_preds = {}, {}, {}

    for player_name in current_roster:
        if player_name in injured_players:
            continue

        gen = general_logs.get(player_name, [])
        if len(gen) >= MIN_FORM_GAMES:
            pred = make_prediction(gen, def_factor, full_sample=12,
                                   rest_factor=rest_factor, pts_boost=pts_boost)
            if pred:
                form_preds[player_name] = pred

        h2h = h2h_logs.get(player_name, [])
        if len(h2h) >= MIN_H2H_GAMES:
            pred = make_prediction(h2h, 1.0, full_sample=4)
            if pred:
                h2h_preds[player_name] = pred

        if player_name in form_preds:
            best_preds[player_name] = blend_predictions(
                form_preds[player_name], h2h_preds.get(player_name), rest_factor, pts_boost)

    by_pts = lambda d: dict(sorted(d.items(), key=lambda x: x[1]['points'], reverse=True))
    form_preds, h2h_preds, best_preds = by_pts(form_preds), by_pts(h2h_preds), by_pts(best_preds)

    print(f"  ✅ {len(form_preds)} form predictions, {len(h2h_preds)} h2h predictions")
    return form_preds, h2h_preds, best_preds

# ─────────────────────────────────────────────────────────────
# 8. OVER/UNDER — separate form + h2h totals
# ─────────────────────────────────────────────────────────────

def predict_totals(home_espn_id, away_espn_id, home_abbrev, away_abbrev, league_avg,
                   home_rest=None, away_rest=None):
    """
    form_total: each team's recent scoring avg × opponent's defensive factor
    h2h_total:  actual scores from past matchups (this season + last)
    """
    home_espn_id, away_espn_id = str(home_espn_id), str(away_espn_id)

    away_scored = _pts_scored.get(away_espn_id)
    home_scored = _pts_scored.get(home_espn_id)

    form_result = None
    if away_scored is not None and home_scored is not None:
        home_def      = get_defensive_factor(home_espn_id, league_avg)
        away_def      = get_defensive_factor(away_espn_id, league_avg)
        home_rest = home_rest or {'factor': 1.0}
        away_rest = away_rest or {'factor': 1.0}
        away_pts_pred = round(away_scored * home_def * away_rest['factor'] * tired_opponent_boost(home_rest), 1)
        home_pts_pred = round(home_scored * away_def * home_rest['factor'] * tired_opponent_boost(away_rest), 1)
        form_result = {
            'predicted_total': round(away_pts_pred + home_pts_pred, 1),
            'away_pts_pred':   away_pts_pred,
            'home_pts_pred':   home_pts_pred,
        }

    # H2H straight from the schedule feed (no extra requests needed).
    history = get_team_history(home_espn_id, home_abbrev)
    h2h_games = [g for g in history
                 if g['opponent_id'] == away_espn_id
                 and g['pts_for'] is not None and g['pts_against'] is not None][:H2H_BACK]

    # Fallback: schedule had no scores -> read them from boxscores.
    if not h2h_games:
        for gid in get_h2h_game_ids(home_espn_id, home_abbrev, away_espn_id):
            scores = parse_boxscore(gid).get('scores', {})
            if len(scores) == 2 and home_espn_id in scores:
                other = [v for k, v in scores.items() if k != home_espn_id][0]
                h2h_games.append({'pts_for': scores[home_espn_id], 'pts_against': other})

    h2h_result = None
    if h2h_games:
        home_pts = [g['pts_for'] for g in h2h_games]
        away_pts = [g['pts_against'] for g in h2h_games]
        totals   = [h + a for h, a in zip(home_pts, away_pts)]
        h2h_result = {
            'predicted_total': round(weighted_average(totals,   0.90), 1),
            'away_pts_pred':   round(weighted_average(away_pts, 0.90), 1),
            'home_pts_pred':   round(weighted_average(home_pts, 0.90), 1),
            'h2h_games_used':  int(len(totals)),
        }
        print(f"  📊 H2H total: {h2h_result['predicted_total']} pts over {len(totals)} games")

    return form_result, h2h_result

# ─────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────

def main():
    games = get_todays_games()

    if not games:
        print("❌ No games today.")
        with open('predictions.json', 'w') as f:
            json.dump({'generated_at': datetime.now().isoformat(), 'season': SEASON_LABEL, 'games': []}, f, indent=2)
        return

    injured_players = get_injured_players()

    all_team_ids = []
    seen = set()
    for g in games:
        for tid, abbrev in [(g['home_espn_id'], g['home']), (g['away_espn_id'], g['away'])]:
            if tid not in seen:
                all_team_ids.append((tid, abbrev))
                seen.add(tid)

    league_avg = build_team_ratings(all_team_ids)

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

        home_rest = get_rest_info(home_id, home, game.get('time', ''))
        away_rest = get_rest_info(away_id, away, game.get('time', ''))
        for ab, r in [(away, away_rest), (home, home_rest)]:
            tag = 'BACK-TO-BACK' if r['back_to_back'] else ('3 games in 4 nights' if r['three_in_four'] else '')
            print(f"  😴 {ab}: {r['rest_days']} rest day(s) {tag} → ×{r['factor']}")

        print(f"\n  👥 {away} (away)...")
        away_form, away_h2h, away_best = process_team(
            away_id, away, home_id, league_avg, injured_players,
            rest_factor=away_rest['factor'], pts_boost=tired_opponent_boost(home_rest))

        print(f"\n  👥 {home} (home)...")
        home_form, home_h2h, home_best = process_team(
            home_id, home, away_id, league_avg, injured_players,
            rest_factor=home_rest['factor'], pts_boost=tired_opponent_boost(away_rest))

        form_total, h2h_total = predict_totals(home_id, away_id, home, away, league_avg,
                                               home_rest, away_rest)

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
            # NEW: form blended with H2H (extra keys; old ones unchanged)
            'home_best':        home_best,
            'away_best':        away_best,
            # NEW: rest days / back-to-back info used in the form + best numbers
            'home_rest':        home_rest,
            'away_rest':        away_rest,
        })

    output = {
        'generated_at': datetime.now().isoformat(),
        'season':       SEASON_LABEL,
        'games':        all_games_output
    }

    with open('predictions.json', 'w') as f:
        json.dump(output, f, indent=2)

    total_form = sum(len(g['home_form']) + len(g['away_form']) for g in all_games_output)
    total_h2h  = sum(len(g['home_h2h'])  + len(g['away_h2h'])  for g in all_games_output)
    print(f"\n✅ Done — {len(all_games_output)} games")
    print(f"   Form predictions: {total_form} players")
    print(f"   H2H  predictions: {total_h2h} players")
    if _failed_requests[0]:
        print(f"   ⚠️  {_failed_requests[0]} ESPN request(s) failed — see warnings above")
    print(f"   → predictions.json")


if __name__ == "__main__":
    main()
