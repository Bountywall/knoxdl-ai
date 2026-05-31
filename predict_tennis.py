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

print("🎾 KnoxDL Tennis Predictor v1 — Form + H2H + Surface")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
    'Referer': 'https://www.espn.com/tennis/'
}

# How many recent matches to look back for form
FORM_BACK = 15
H2H_BACK  = 10

# ESPN tennis uses two different base URLs
SITE_API = "https://site.web.api.espn.com/apis/site/v2/sports/tennis"
CORE_API = "http://sports.core.api.espn.com/v2/sports/tennis"

YEAR = datetime.now().year

_last_req = [time.time()]

def rate_limit(mn=0.6, mx=1.4):
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
        return None

def follow_ref(ref_url):
    """Follow a $ref URL directly."""
    rate_limit()
    try:
        r = SESSION.get(ref_url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.json()
    except:
        return None

# ─────────────────────────────────────────────────────────────
# 1. TODAY'S MATCHES (ATP + WTA)
# ─────────────────────────────────────────────────────────────

def get_todays_matches():
    print("\n📅 Fetching today's matches (ATP + WTA)...")
    today = datetime.now().strftime('%Y%m%d')
    matches = []

    for league in ['atp', 'wta']:
        data = espn_get(f"{SITE_API}/{league}/scoreboard", params={'dates': today})
        if not data:
            continue

        for event in data.get('events', []):
            event_id   = event.get('id', '')
            event_name = event.get('name', 'Unknown Tournament')
            comp = event.get('competitions', [{}])[0]
            competitors = comp.get('competitors', [])
            if len(competitors) < 2:
                continue

            status = event.get('status', {}).get('type', {}).get('description', 'Scheduled')

            p1 = competitors[0]
            p2 = competitors[1]

            def extract_athlete(c):
                a = c.get('athlete', {})
                return {
                    'id':      str(a.get('id', c.get('id', ''))),
                    'name':    a.get('displayName', a.get('fullName', 'Unknown')),
                    'seed':    c.get('seeding', None),
                    'winner':  c.get('winner', False),
                }

            a1 = extract_athlete(p1)
            a2 = extract_athlete(p2)

            if not a1['id'] or not a2['id']:
                continue

            # Parse surface from competition details (court field)
            surface = comp.get('court', {}).get('surface', {}).get('type', '') if isinstance(comp.get('court'), dict) else ''

            matches.append({
                'match_label':  f"{a1['name']} vs {a2['name']}",
                'tournament':   event_name,
                'event_id':     event_id,
                'comp_id':      comp.get('id', ''),
                'league':       league,
                'surface':      surface,
                'player1':      a1,
                'player2':      a2,
                'status':       status,
                'match_time':   event.get('date', ''),
            })

    # Prefer scheduled over final
    upcoming = [m for m in matches if any(x in m['status'] for x in ['Scheduled', 'PM', 'AM', 'ET', 'PT', 'In Progress'])]
    result = upcoming if upcoming else matches
    print(f"✅ {len(result)} match(es) found today")
    return result

# ─────────────────────────────────────────────────────────────
# 2. RANKINGS (used as confidence weight)
# ─────────────────────────────────────────────────────────────

_rankings = {}  # athlete_id -> rank

def load_rankings():
    print("\n🏆 Loading ATP + WTA rankings...")
    for league in ['atp', 'wta']:
        data = espn_get(f"{SITE_API}/{league}/rankings")
        if not data:
            continue
        for ranking_group in data.get('rankings', []):
            for entry in ranking_group.get('ranks', []):
                athlete = entry.get('athlete', {})
                aid     = str(athlete.get('id', ''))
                rank    = entry.get('current', 999)
                if aid:
                    _rankings[aid] = rank
    print(f"✅ {len(_rankings)} players ranked")

def get_rank(athlete_id):
    return _rankings.get(str(athlete_id), 999)

# ─────────────────────────────────────────────────────────────
# 3. ATHLETE PROFILE
# ─────────────────────────────────────────────────────────────

_athlete_cache = {}

def get_athlete_profile(athlete_id):
    aid = str(athlete_id)
    if aid in _athlete_cache:
        return _athlete_cache[aid]

    data = espn_get(f"{CORE_API}/athletes/{aid}")
    if not data:
        _athlete_cache[aid] = {}
        return {}

    profile = {
        'id':        aid,
        'name':      data.get('displayName', ''),
        'hand':      data.get('hand', {}).get('type', '') if isinstance(data.get('hand'), dict) else '',
        'country':   data.get('citizenshipCountry', {}).get('name', '') if isinstance(data.get('citizenshipCountry'), dict) else '',
        'age':       data.get('age', None),
    }

    # Season wins/losses from statistics endpoint
    stats_data = espn_get(f"{CORE_API}/athletes/{aid}/statistics")
    if stats_data:
        for cat in stats_data.get('splits', {}).get('categories', stats_data.get('categories', [])):
            for s in cat.get('stats', []):
                name = s.get('name', '')
                val  = s.get('value', 0)
                if name == 'singlesWon':
                    profile['season_wins'] = int(val)
                elif name == 'singlesLost':
                    profile['season_losses'] = int(val)
                elif name == 'singlesTitles':
                    profile['titles'] = int(val)
                elif name == 'prize':
                    profile['prize_money'] = int(val)

    _athlete_cache[aid] = profile
    return profile

# ─────────────────────────────────────────────────────────────
# 4. EVENTLOG — get all recent match refs for a player
# ─────────────────────────────────────────────────────────────

_eventlog_cache = {}

def get_athlete_eventlog(athlete_id):
    """Returns list of { competition_ref, competitor_ref, event_ref } dicts."""
    aid = str(athlete_id)
    if aid in _eventlog_cache:
        return _eventlog_cache[aid]

    data = espn_get(f"{CORE_API}/athletes/{aid}/eventlog")
    if not data:
        _eventlog_cache[aid] = []
        return []

    items = data.get('events', {}).get('items', [])
    result = []
    for item in items:
        if not item.get('played', True):
            continue
        comp_ref = item.get('competition', {}).get('$ref', '')
        cptr_ref = item.get('competitor',  {}).get('$ref', '')
        evt_ref  = item.get('event',       {}).get('$ref', '')
        if comp_ref:
            result.append({
                'competition_ref': comp_ref,
                'competitor_ref':  cptr_ref,
                'event_ref':       evt_ref,
            })

    _eventlog_cache[aid] = result
    print(f"  📋 {aid}: {len(result)} logged match(es)")
    return result

# ─────────────────────────────────────────────────────────────
# 5. PARSE A SINGLE MATCH — result + surface
# ─────────────────────────────────────────────────────────────

_match_cache = {}

def parse_match(competition_ref, competitor_ref, my_athlete_id):
    """
    Returns dict:
      won, sets_won, sets_lost, surface, opponent_id, opponent_rank, tournament_name
    or None if unavailable.
    """
    cache_key = competition_ref
    if cache_key in _match_cache:
        cached = _match_cache[cache_key]
        if cached is None:
            return None
        # Return perspective for this athlete
        return _perspective(cached, str(my_athlete_id))

    comp_data = follow_ref(competition_ref)
    if not comp_data:
        _match_cache[cache_key] = None
        return None

    status = comp_data.get('status', {}).get('type', {}).get('name', '')
    if status != 'STATUS_FINAL':
        _match_cache[cache_key] = None
        return None

    # Surface
    court   = comp_data.get('court', {})
    surface = court.get('surface', {}).get('type', '') if isinstance(court, dict) else ''

    # Competitors
    competitors = comp_data.get('competitors', [])
    players = {}
    for c in competitors:
        cid     = str(c.get('id', ''))
        winner  = c.get('winner', False)
        score   = c.get('score', '')       # e.g. "6-4, 7-5"
        linescores = c.get('linescores', [])
        sets_won  = sum(1 for ls in linescores if ls.get('winner', False))
        sets_lost = len(linescores) - sets_won
        players[cid] = {
            'winner':     winner,
            'score':      score,
            'sets_won':   sets_won,
            'sets_lost':  sets_lost,
        }

    # Tournament name from event ref if possible
    tournament = comp_data.get('notes', [{}])[0].get('headline', '') if comp_data.get('notes') else ''

    record = {
        'surface':     surface,
        'tournament':  tournament,
        'players':     players,
        'round':       comp_data.get('round', {}).get('displayName', ''),
    }
    _match_cache[cache_key] = record
    return _perspective(record, str(my_athlete_id))

def _perspective(record, my_id):
    """Return the match from one player's point of view."""
    if record is None:
        return None
    players = record.get('players', {})
    me  = players.get(my_id)
    opp_id = next((k for k in players if k != my_id), None)
    opp = players.get(opp_id, {}) if opp_id else {}

    if me is None:
        return None

    return {
        'won':           me.get('winner', False),
        'sets_won':      me.get('sets_won', 0),
        'sets_lost':     opp.get('sets_won', 0),   # opponent's sets won = my sets lost
        'surface':       record.get('surface', ''),
        'round':         record.get('round', ''),
        'tournament':    record.get('tournament', ''),
        'opponent_id':   opp_id,
        'opponent_rank': get_rank(opp_id) if opp_id else 999,
    }

# ─────────────────────────────────────────────────────────────
# 6. BUILD MATCH LOGS FOR A PLAYER
# ─────────────────────────────────────────────────────────────

def build_match_logs(athlete_id, limit=FORM_BACK):
    """Returns list of match perspective dicts, most recent first."""
    logs_raw = get_athlete_eventlog(athlete_id)
    logs = []
    for item in logs_raw[:limit]:
        result = parse_match(item['competition_ref'], item['competitor_ref'], athlete_id)
        if result:
            logs.append(result)
    return logs

def build_h2h_logs(athlete_id, opponent_id, limit=H2H_BACK):
    """Returns only matches where opponent_id was the opponent."""
    all_logs = get_athlete_eventlog(athlete_id)
    h2h = []
    for item in all_logs:
        result = parse_match(item['competition_ref'], item['competitor_ref'], athlete_id)
        if result and str(result.get('opponent_id', '')) == str(opponent_id):
            h2h.append(result)
            if len(h2h) >= limit:
                break
    return h2h

# ─────────────────────────────────────────────────────────────
# 7. PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────

def weighted_win_rate(logs, decay=0.90):
    """Exponentially weighted win rate — recent matches weighted more."""
    if not logs:
        return 0.5
    n       = len(logs)
    weights = np.array([decay ** i for i in range(n)])
    weights /= weights.sum()
    wins    = np.array([1.0 if m['won'] else 0.0 for m in logs])
    return float(np.dot(weights, wins))

def surface_win_rate(logs, surface):
    """Win rate on a specific surface."""
    surface_logs = [m for m in logs if m.get('surface', '').lower() == surface.lower()]
    if len(surface_logs) < 2:
        return None  # not enough data
    wins = sum(1 for m in surface_logs if m['won'])
    return wins / len(surface_logs)

def avg_sets_ratio(logs):
    """Average sets won per match (gives a feel for how dominant)."""
    if not logs:
        return 0.5
    ratios = []
    for m in logs:
        total = m['sets_won'] + m['sets_lost']
        if total > 0:
            ratios.append(m['sets_won'] / total)
    return float(np.mean(ratios)) if ratios else 0.5

def rank_factor(my_rank, opp_rank):
    """
    Returns a float near 1.0 adjusted by ranking difference.
    Better rank (lower number) vs worse opponent → slight boost.
    """
    if my_rank == 999 and opp_rank == 999:
        return 1.0
    # Use log scale so rank 1 vs 50 isn't wildly different from 50 vs 100
    my_score  = 1.0 / max(my_rank,  1)
    opp_score = 1.0 / max(opp_rank, 1)
    total = my_score + opp_score
    if total == 0:
        return 0.5
    raw = my_score / total  # 0–1 range
    # Compress toward 0.5 so rank alone isn't too decisive
    return float(0.35 + raw * 0.30)

def predict_match(p1_id, p2_id, surface=''):
    """
    Returns dict with form-based and h2h-based win probability for p1.
    """
    print(f"\n  🔮 Predicting: id={p1_id} vs id={p2_id} | surface={surface or 'unknown'}")

    p1_rank = get_rank(p1_id)
    p2_rank = get_rank(p2_id)
    print(f"     Ranks: p1={p1_rank}  p2={p2_rank}")

    # ── Form logs ──
    print(f"     Fetching form logs for p1...")
    p1_logs = build_match_logs(p1_id)
    print(f"     Fetching form logs for p2...")
    p2_logs = build_match_logs(p2_id)

    # ── H2H logs ──
    print(f"     Fetching H2H logs...")
    h2h_p1 = build_h2h_logs(p1_id, p2_id)
    h2h_p2 = build_h2h_logs(p2_id, p1_id)

    # ── Form-based prediction ──
    p1_form_wr  = weighted_win_rate(p1_logs)
    p2_form_wr  = weighted_win_rate(p2_logs)
    p1_sets_avg = avg_sets_ratio(p1_logs)
    p2_sets_avg = avg_sets_ratio(p2_logs)

    # Surface adjustment
    p1_surf_wr = surface_win_rate(p1_logs, surface) if surface else None
    p2_surf_wr = surface_win_rate(p2_logs, surface) if surface else None

    # Blend general form + surface form (60/40 if surface data exists)
    p1_eff = p1_form_wr if p1_surf_wr is None else (0.60 * p1_form_wr + 0.40 * p1_surf_wr)
    p2_eff = p2_form_wr if p2_surf_wr is None else (0.60 * p2_form_wr + 0.40 * p2_surf_wr)

    # Rank factor
    rf_p1  = rank_factor(p1_rank, p2_rank)
    rf_p2  = rank_factor(p2_rank, p1_rank)

    # Combine: 60% form, 40% rank
    p1_combined = 0.60 * p1_eff + 0.40 * rf_p1
    p2_combined = 0.60 * p2_eff + 0.40 * rf_p2

    # Normalize to probability
    total = p1_combined + p2_combined
    form_p1_prob = p1_combined / total if total > 0 else 0.5

    form_result = {
        'p1_win_prob':    round(form_p1_prob,   3),
        'p2_win_prob':    round(1 - form_p1_prob, 3),
        'p1_form_wr':     round(p1_form_wr, 3),
        'p2_form_wr':     round(p2_form_wr, 3),
        'p1_surface_wr':  round(p1_surf_wr, 3) if p1_surf_wr is not None else None,
        'p2_surface_wr':  round(p2_surf_wr, 3) if p2_surf_wr is not None else None,
        'p1_sets_avg':    round(p1_sets_avg, 3),
        'p2_sets_avg':    round(p2_sets_avg, 3),
        'p1_rank':        p1_rank,
        'p2_rank':        p2_rank,
        'p1_matches_analyzed': len(p1_logs),
        'p2_matches_analyzed': len(p2_logs),
        'surface':        surface,
    }

    # ── H2H-based prediction ──
    h2h_result = None
    if h2h_p1 or h2h_p2:
        h2h_n       = len(h2h_p1)
        h2h_p1_wins = sum(1 for m in h2h_p1 if m['won'])
        h2h_p2_wins = h2h_n - h2h_p1_wins

        # Also get weighted rate
        h2h_wr_p1 = weighted_win_rate(h2h_p1) if h2h_p1 else 0.5
        h2h_wr_p2 = 1 - h2h_wr_p1

        h2h_result = {
            'p1_win_prob':  round(h2h_wr_p1, 3),
            'p2_win_prob':  round(h2h_wr_p2, 3),
            'p1_h2h_wins':  h2h_p1_wins,
            'p2_h2h_wins':  h2h_p2_wins,
            'h2h_matches':  h2h_n,
        }
        print(f"     H2H: p1 {h2h_p1_wins}W - {h2h_p2_wins}W p2 ({h2h_n} matches)")

    # ── Confidence ──
    # Based on sample sizes
    sample_score = min((len(p1_logs) + len(p2_logs)) / (2 * FORM_BACK), 1.0)
    confidence   = int(round(sample_score * 100))

    # ── Predicted sets ──
    # Estimate most likely scoreline based on win prob
    p1_prob = form_p1_prob
    sets_format = 3   # best of 3 (most ATP/WTA except slams)
    if p1_prob >= 0.65:
        predicted_score = "2-0"
    elif p1_prob >= 0.52:
        predicted_score = "2-1"
    elif p1_prob >= 0.48:
        predicted_score = "2-1 (close)"
    elif p1_prob >= 0.35:
        predicted_score = "1-2"
    else:
        predicted_score = "0-2"

    return {
        'form':            form_result,
        'h2h':             h2h_result,
        'confidence':      confidence,
        'predicted_score': predicted_score,
    }

# ─────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────

def main():
    matches = get_todays_matches()

    if not matches:
        print("❌ No matches today.")
        with open('predictions.json', 'w') as f:
            json.dump({
                'generated_at': datetime.now().isoformat(),
                'season':       str(YEAR),
                'sport':        'tennis',
                'matches':      []
            }, f, indent=2)
        return

    load_rankings()

    all_output = []

    for i, match in enumerate(matches):
        p1 = match['player1']
        p2 = match['player2']
        label = match['match_label']

        print(f"\n{'='*60}")
        print(f"🎾 Match {i+1}/{len(matches)}: {label}")
        print(f"   Tournament: {match['tournament']} | Surface: {match['surface'] or 'unknown'} | League: {match['league'].upper()}")
        print(f"{'='*60}")

        # Fetch profiles
        p1_profile = get_athlete_profile(p1['id'])
        p2_profile = get_athlete_profile(p2['id'])

        prediction = predict_match(p1['id'], p2['id'], surface=match.get('surface', ''))

        all_output.append({
            'match':          label,
            'tournament':     match['tournament'],
            'league':         match['league'].upper(),
            'surface':        match['surface'],
            'match_time':     match['match_time'],
            'status':         match['status'],
            'player1': {
                **p1,
                'rank':         get_rank(p1['id']),
                'season_wins':  p1_profile.get('season_wins',   0),
                'season_losses':p1_profile.get('season_losses', 0),
                'titles':       p1_profile.get('titles',        0),
                'country':      p1_profile.get('country',       ''),
                'hand':         p1_profile.get('hand',          ''),
            },
            'player2': {
                **p2,
                'rank':         get_rank(p2['id']),
                'season_wins':  p2_profile.get('season_wins',   0),
                'season_losses':p2_profile.get('season_losses', 0),
                'titles':       p2_profile.get('titles',        0),
                'country':      p2_profile.get('country',       ''),
                'hand':         p2_profile.get('hand',          ''),
            },
            'prediction':     prediction,
        })

    output = {
        'generated_at': datetime.now().isoformat(),
        'season':       str(YEAR),
        'sport':        'tennis',
        'matches':      all_output,
    }

    with open('predictions.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Done — {len(all_output)} match(es) predicted")
    print("   → predictions.json")

if __name__ == "__main__":
    main()
