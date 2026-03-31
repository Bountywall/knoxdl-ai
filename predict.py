import asyncio
import aiohttp
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
from datetime import datetime
from nba_api.stats.endpoints import commonteamroster, playergamelog
from nba_api.stats.static import teams
import nest_asyncio
import time
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
import json
import os

warnings.filterwarnings('ignore')
nest_asyncio.apply()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class HybridNBAPredictor:
    def __init__(self):
        self.season = self.get_current_season()
        self.scaler = StandardScaler()
        self.model = None
        self.request_count = 0
        self.last_request_time = time.time()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.espn.com/'
        }

    def get_current_season(self):
        now = datetime.now()
        return now.year if now.month >= 10 else now.year - 1

    def create_session_with_retries(self):
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def rate_limit(self, min_delay=1.0, max_delay=3.0):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < min_delay:
            sleep_time = random.uniform(min_delay, max_delay)
            time.sleep(sleep_time)
        self.request_count += 1
        self.last_request_time = time.time()

    def get_todays_games(self):
        print("📅 Fetching today's NBA games from ESPN API...")
        try:
            url = "https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
            params = {
                'dates': datetime.now().strftime('%Y%m%d'),
                'region': 'us',
                'lang': 'en',
                'contentorigin': 'espn'
            }
            self.rate_limit(min_delay=2.0, max_delay=4.0)
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            data = response.json()

            games = []
            for event in data.get('events', []):
                if 'competitions' in event and event['competitions']:
                    competition = event['competitions'][0]
                    if len(competition['competitors']) >= 2:
                        away_team = competition['competitors'][0]['team']['abbreviation']
                        home_team = competition['competitors'][1]['team']['abbreviation']
                        games.append({
                            'home': home_team,
                            'away': away_team,
                            'game_id': event['id'],
                            'time': event.get('date', ''),
                            'status': event.get('status', {}).get('type', {}).get('description', 'Scheduled')
                        })

            if games:
                print(f"✅ Found {len(games)} games via ESPN API")
                upcoming_games = [g for g in games if any(x in g['status'] for x in ['Scheduled', 'PM', 'AM', 'ET', 'PT'])]
                return upcoming_games if upcoming_games else games
            else:
                print("❌ No games found via ESPN API")
                return []
        except Exception as e:
            print(f"❌ ESPN API failed: {e}")
            return []

    def get_team_roster(self, team_abbrev):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.rate_limit(min_delay=2.0, max_delay=5.0)
                print(f"🔍 Getting roster for {team_abbrev} (attempt {attempt + 1})...")
                nba_teams = teams.get_teams()
                team = [t for t in nba_teams if t['abbreviation'] == team_abbrev]
                if not team:
                    print(f"❌ Team {team_abbrev} not found in NBA API")
                    return []
                team_id = team[0]['id']
                session = self.create_session_with_retries()
                roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=self.season, timeout=60)
                roster_data = roster.get_dict()
                players = []
                for player in roster_data['resultSets'][0]['rowSet']:
                    players.append({
                        'name': player[3],
                        'player_id': player[14],
                        'team': team_abbrev,
                        'position': player[5]
                    })
                print(f"✅ Found {len(players)} players for {team_abbrev}")
                return players
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1} failed for {team_abbrev}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(5.0, 10.0))
                else:
                    return []

    async def get_player_game_logs(self, session, player_id, player_name):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(random.uniform(2.0, 4.0))
                print(f"📊 Getting stats for {player_name} (attempt {attempt + 1})...")
                gamelog = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=self.season,
                    timeout=60
                )
                data = gamelog.get_dict()
                if not data['resultSets'] or not data['resultSets'][0]['rowSet']:
                    print(f"⚠️ No game logs found for {player_name}")
                    return None
                logs = []
                for game in data['resultSets'][0]['rowSet']:
                    log = {
                        'PTS': game[24] if game[24] is not None else 0,
                        'REB': game[18] if game[18] is not None else 0,
                        'AST': game[19] if game[19] is not None else 0,
                        'STL': game[20] if game[20] is not None else 0,
                        'BLK': game[21] if game[21] is not None else 0,
                        'TOV': game[22] if game[22] is not None else 0,
                        'MIN': self.parse_minutes(game[7]) if game[7] is not None else 0
                    }
                    logs.append(log)
                df = pd.DataFrame(logs).head(15)
                if len(df) > 0:
                    print(f"✅ Got {len(df)} recent games for {player_name}")
                return df
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1} failed for {player_name}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(random.uniform(8.0, 15.0))
                else:
                    return None

    def parse_minutes(self, minutes_str):
        try:
            if minutes_str is None:
                return 0.0
            if ':' in minutes_str:
                parts = minutes_str.split(':')
                return float(parts[0])
            return float(minutes_str)
        except:
            return 0.0

    def build_model(self, input_size):
        return nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        ).to(device)

    def predict_stats(self, game_logs, player_name):
        if game_logs is None or game_logs.empty:
            print(f"    ❌ No game logs for {player_name}")
            return self.get_fallback_prediction()
        recent = game_logs.head(10)
        if len(recent) == 0:
            return self.get_fallback_prediction()
        predictions = {
            'points':         round(recent['PTS'].mean(), 1),
            'rebounds':       round(recent['REB'].mean(), 1),
            'assists':        round(recent['AST'].mean(), 1),
            'steals':         round(recent['STL'].mean(), 1),
            'blocks':         round(recent['BLK'].mean(), 1),
            'turnovers':      round(recent['TOV'].mean(), 1),
            'minutes':        round(recent['MIN'].mean(), 1),
            'games_analyzed': len(recent)
        }
        print(f"    📈 {player_name}: {predictions['points']} PTS, {predictions['rebounds']} REB, {predictions['minutes']} MIN")
        return predictions

    def get_fallback_prediction(self):
        return {
            'points': 0.0, 'rebounds': 0.0, 'assists': 0.0,
            'steals': 0.0, 'blocks': 0.0, 'turnovers': 0.0,
            'minutes': 0.0, 'games_analyzed': 0
        }


async def process_team_players(session, predictor, team, team_side, game_predictions):
    roster = predictor.get_team_roster(team)
    if not roster:
        print(f"❌ Could not load roster for {team}")
        return

    print(f"👥 Processing {len(roster)} players for {team}")
    batch_size = 3
    successful_players = 0

    for i in range(0, len(roster), batch_size):
        batch = roster[i:i + batch_size]
        tasks = [predictor.get_player_game_logs(session, p['player_id'], p['name']) for p in batch]
        try:
            logs_list = await asyncio.gather(*tasks, return_exceptions=True)
            for player, logs in zip(batch, logs_list):
                if isinstance(logs, Exception):
                    print(f"❌ Error processing {player['name']}: {logs}")
                    continue
                predictions = predictor.predict_stats(logs, player['name'])
                if predictions['games_analyzed'] > 0:
                    game_predictions[team_side][player['name']] = predictions
                    successful_players += 1
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")

        if i + batch_size < len(roster):
            delay = random.uniform(8.0, 15.0)
            print(f"⏳ Waiting {delay:.1f}s before next batch...")
            await asyncio.sleep(delay)

    print(f"🎯 Successfully processed {successful_players}/{len(roster)} players for {team}")


async def main():
    predictor = HybridNBAPredictor()

    print("🏀 KnoxDL NBA Player Props Predictor")
    print("=" * 50)

    games = predictor.get_todays_games()
    if not games:
        print("No NBA games found today")
        # Write empty predictions.json so the site shows "no games today"
        output = {
            'generated_at': datetime.now().isoformat(),
            'season': str(predictor.season),
            'games': []
        }
        with open('predictions.json', 'w') as f:
            json.dump(output, f, indent=2)
        return

    print(f"🎯 Targeting {len(games)} game(s)")
    all_predictions = {}

    async with aiohttp.ClientSession() as session:
        for i, game in enumerate(games):
            home_team = game['home']
            away_team = game['away']
            game_key  = f"{away_team} @ {home_team}"

            print(f"\n{'='*50}")
            print(f"🎯 Processing Game {i+1}/{len(games)}: {game_key}")
            print(f"{'='*50}")

            game_predictions = {'home': {}, 'away': {}}

            await process_team_players(session, predictor, away_team, 'away', game_predictions)

            team_delay = random.uniform(15.0, 25.0)
            print(f"⏳ Waiting {team_delay:.1f}s before processing home team...")
            await asyncio.sleep(team_delay)

            await process_team_players(session, predictor, home_team, 'home', game_predictions)
            all_predictions[game_key] = game_predictions

            if i < len(games) - 1:
                game_delay = random.uniform(30.0, 45.0)
                print(f"⏳ Long delay {game_delay:.1f}s before next game...")
                await asyncio.sleep(game_delay)

    # ── Build structured output ────────────────────────────────
    games_output = []
    for game_key, teams_data in all_predictions.items():
        parts     = game_key.split(' @ ')
        away_team = parts[0] if len(parts) == 2 else ''
        home_team = parts[1] if len(parts) == 2 else ''

        def sorted_players(side):
            players = teams_data.get(side, {})
            return dict(sorted(players.items(), key=lambda x: x[1]['points'], reverse=True))

        # Find game time from original games list
        game_time = next((g['time'] for g in games if g['home'] == home_team and g['away'] == away_team), '')

        games_output.append({
            'game':         game_key,
            'home_team':    home_team,
            'away_team':    away_team,
            'game_time':    game_time,
            'home_players': sorted_players('home'),
            'away_players': sorted_players('away'),
        })

    output = {
        'generated_at': datetime.now().isoformat(),
        'season':       str(predictor.season),
        'games':        games_output
    }

    with open('predictions.json', 'w') as f:
        json.dump(output, f, indent=2)

    total_players = sum(len(g['home_players']) + len(g['away_players']) for g in games_output)
    print(f"\n✅ Done — {len(games_output)} games, {total_players} players → predictions.json")

if __name__ == "__main__":
    asyncio.run(main())
