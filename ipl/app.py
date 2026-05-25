from flask import Flask, render_template, request, jsonify, make_response
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from functools import lru_cache

app = Flask(__name__)

# ── Load data once ──────────────────────────────────────────────
df = pd.read_csv('deliveries.csv')
df['extras_type'] = df['extras_type'].fillna('none')
df['player_dismissed'] = df['player_dismissed'].fillna('none')
df['dismissal_kind'] = df['dismissal_kind'].fillna('none')
df['fielder'] = df['fielder'].fillna('none')

TEAMS = sorted(df['batting_team'].unique().tolist())
BATTERS = sorted(df['batter'].unique().tolist())
BOWLERS = sorted(df['bowler'].unique().tolist())

# ── Load encoders & models ──────────────────────────────────────
with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

def load_model(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except:
        return None

wicket_model  = load_model('wicket_model.pkl')
runs_model    = load_model('runs_model.pkl')
score_model   = load_model('score_model.pkl')
matchup_model = load_model('matchup_model.pkl')

with open('matchup_encoders.pkl', 'rb') as f:
    matchup_encoders = pickle.load(f)

# ── Helper ──────────────────────────────────────────────────────
def encode(col, val):
    try:
        return int(encoders[col].transform([val])[0])
    except:
        return 0

def encode_matchup(col, val):
    try:
        return int(matchup_encoders[col].transform([val])[0])
    except:
        return 0

# ── Routes ──────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', teams=TEAMS, batters=BATTERS, bowlers=BOWLERS)

@app.route('/predict/wicket', methods=['POST'])
def predict_wicket():
    d = request.json
    features = [[
        int(d['inning']), encode('batting_team', d['batting_team']),
        encode('bowling_team', d['bowling_team']), int(d['over']),
        int(d['ball']), encode('batter', d['batter']),
        encode('bowler', d['bowler']), float(d['cum_runs']),
        float(d['cum_wickets']), float(d['balls_bowled']), 0
    ]]
    prob = wicket_model.predict_proba(features)[0][1]
    return jsonify({'probability': round(float(prob) * 100, 2), 'prediction': 'Wicket' if prob > 0.5 else 'No Wicket'})

@app.route('/predict/runs', methods=['POST'])
def predict_runs():
    d = request.json
    features = [[
        int(d['inning']), encode('batting_team', d['batting_team']),
        encode('bowling_team', d['bowling_team']), int(d['over']),
        int(d['ball']), encode('batter', d['batter']),
        encode('bowler', d['bowler']), float(d['cum_runs']),
        float(d['cum_wickets']), float(d['balls_bowled']), 0
    ]]
    pred = int(runs_model.predict(features)[0])
    probs = runs_model.predict_proba(features)[0]
    classes = runs_model.classes_.tolist()
    dist = {str(c): round(float(p) * 100, 1) for c, p in zip(classes, probs)}
    return jsonify({'predicted_runs': pred, 'distribution': dist})

@app.route('/predict/score', methods=['POST'])
def predict_score():
    d = request.json
    features = [[
        encode('batting_team', d['batting_team']),
        encode('bowling_team', d['bowling_team']),
        int(d['over']), int(d['ball']),
        float(d['cum_runs']), float(d['cum_wickets']),
        float(d['balls_bowled'])
    ]]
    pred = float(score_model.predict(features)[0])
    return jsonify({'predicted_score': round(pred, 1)})

@app.route('/predict/matchup', methods=['POST'])
def predict_matchup():
    d = request.json
    batter_enc = encode_matchup('batter', d['batter'])
    bowler_enc = encode_matchup('bowler', d['bowler'])
    features = [[batter_enc, bowler_enc, int(d['balls']), 0]]
    pred = float(matchup_model.predict(features)[0])

    # Historical stats
    hist = df[(df['batter'] == d['batter']) & (df['bowler'] == d['bowler'])]
    stats = {}
    if len(hist) > 0:
        stats = {
            'balls': int(len(hist)),
            'runs': int(hist['batsman_runs'].sum()),
            'dismissals': int(hist['is_wicket'].sum()),
            'strike_rate': round(hist['batsman_runs'].sum() / len(hist) * 100, 1)
        }
    return jsonify({'predicted_runs': round(pred, 1), 'historical': stats})

@lru_cache(maxsize=256)
def _batter_analysis(batter):
    stats = df[df['batter'] == batter]
    if len(stats) == 0:
        return {'error': 'Batter not found'}
    balls = int((stats['extra_runs'] == 0).sum())
    runs = int(stats['batsman_runs'].sum())
    fours = int((stats['batsman_runs'] == 4).sum())
    sixes = int((stats['batsman_runs'] == 6).sum())
    dismissals = int(df[df['player_dismissed'] == batter]['is_wicket'].sum())
    strike_rate = round(runs / balls * 100, 2) if balls > 0 else 0
    avg = round(runs / dismissals, 2) if dismissals > 0 else runs
    dot_pct = round(float((stats['batsman_runs'] == 0).sum()) / len(stats) * 100, 1)
    over_stats = stats.groupby('over').agg(
        runs=('batsman_runs', 'sum'), balls=('batsman_runs', 'count')
    ).reset_index()
    over_stats['sr'] = (over_stats['runs'] / over_stats['balls'] * 100).round(1)
    return {
        'batter': batter, 'runs': runs, 'balls': balls,
        'fours': fours, 'sixes': sixes, 'dismissals': dismissals,
        'strike_rate': strike_rate, 'average': avg,
        'dot_pct': dot_pct, 'over_breakdown': over_stats.to_dict(orient='records')
    }

@app.route('/analyze/batter', methods=['POST'])
def analyze_batter():
    return jsonify(_batter_analysis(request.json['batter']))

@app.route('/predict/win', methods=['POST'])
def predict_win():
    d = request.json
    team1 = d['team1']
    team2 = d['team2']
    target = int(d['target'])
    mode = d.get('mode', 'first')  # 'first' or 'chase'

    if mode == 'first':
        # Historical win rate for team1 when batting first with similar scores
        margin = 15
        matches = df[df['inning'] == 1].groupby('match_id').agg(
            batting_team=('batting_team', 'first'),
            bowling_team=('bowling_team', 'first'),
            score=('total_runs', 'sum')
        ).reset_index()
        similar = matches[(matches['score'] >= target - margin) & (matches['score'] <= target + margin)]
        if len(similar) == 0:
            similar = matches
        # Find 2nd innings results
        inn2 = df[df['inning'] == 2].groupby('match_id').agg(
            score=('total_runs', 'sum'),
            wickets=('is_wicket', 'sum')
        ).reset_index()
        merged = similar.merge(inn2, on='match_id', suffixes=('_1', '_2'))
        if len(merged) == 0:
            return jsonify({'error': 'Not enough data'})
        batting_first_wins = int((merged['score_1'] > merged['score_2']).sum())
        total = len(merged)
        win_pct = round(batting_first_wins / total * 100, 1)
        return jsonify({
            'team1': team1, 'team2': team2, 'target': target,
            'team1_win_pct': win_pct, 'team2_win_pct': round(100 - win_pct, 1),
            'sample_size': total, 'mode': 'first'
        })
    else:
        # Chase mode: team2 needs target, given current score/overs
        current_score = int(d.get('current_score', 0))
        current_over = float(d.get('current_over', 0))
        wickets_lost = int(d.get('wickets_lost', 0))
        balls_done = int(current_over * 6)
        balls_left = 120 - balls_done
        runs_needed = target - current_score
        if balls_left <= 0 or runs_needed <= 0:
            win = runs_needed <= 0
            return jsonify({'team2_win_pct': 100.0 if win else 0.0, 'team1_win_pct': 0.0 if win else 100.0,
                            'runs_needed': max(0, runs_needed), 'balls_left': 0, 'rrr': 0, 'mode': 'chase'})
        rrr = round(runs_needed / balls_left * 6, 2)
        # Simple logistic-style estimate based on RRR and wickets
        wickets_left = 10 - wickets_lost
        # Higher RRR and fewer wickets = lower win chance
        base = 50
        rrr_factor = (rrr - 6) * 5   # each run above 6/over reduces by 5%
        wkt_factor = (5 - wickets_left) * 3  # fewer wickets reduces
        chase_win_pct = max(5, min(95, base - rrr_factor - wkt_factor))
        return jsonify({
            'team1': team1, 'team2': team2, 'target': target,
            'team2_win_pct': round(chase_win_pct, 1),
            'team1_win_pct': round(100 - chase_win_pct, 1),
            'runs_needed': runs_needed, 'balls_left': balls_left, 'rrr': rrr, 'mode': 'chase'
        })

@lru_cache(maxsize=256)
def _bowler_analysis(bowler):
    stats = df[df['bowler'] == bowler]
    if len(stats) == 0:
        return {'error': 'Bowler not found'}
    balls = len(stats)
    overs = balls / 6
    runs = int(stats['total_runs'].sum())
    wickets = int(stats['is_wicket'].sum())
    economy = round(runs / overs, 2) if overs > 0 else 0
    sr = round(balls / wickets, 1) if wickets > 0 else None
    wicket_prob = round(wickets / balls * 100, 2)
    over_stats = stats.groupby('over').agg(
        runs=('total_runs', 'sum'), wickets=('is_wicket', 'sum'), balls=('ball', 'count')
    ).reset_index()
    return {
        'bowler': bowler, 'balls': balls, 'overs': round(overs, 1),
        'runs_conceded': runs, 'wickets': wickets,
        'economy': economy, 'strike_rate': sr,
        'wicket_probability': wicket_prob, 'over_breakdown': over_stats.to_dict(orient='records')
    }

@app.route('/analyze/bowler', methods=['POST'])
def analyze_bowler():
    return jsonify(_bowler_analysis(request.json['bowler']))

@app.route('/compare', methods=['POST'])
def compare():
    d = request.json
    mode = d['mode']  # 'batter_batter', 'bowler_bowler', 'batter_bowler'

    def batter_stats(name):
        s = df[df['batter'] == name]
        if len(s) == 0: return None
        balls = len(s[s['extra_runs'] == 0])
        runs = int(s['batsman_runs'].sum())
        fours = int((s['batsman_runs'] == 4).sum())
        sixes = int((s['batsman_runs'] == 6).sum())
        dismissals = int(df[df['player_dismissed'] == name]['is_wicket'].sum())
        sr = round(runs / balls * 100, 2) if balls > 0 else 0
        avg = round(runs / dismissals, 2) if dismissals > 0 else runs
        dot_pct = round((s['batsman_runs'] == 0).sum() / len(s) * 100, 1)
        boundary_pct = round((fours + sixes) / len(s) * 100, 1)
        return {'name': name, 'runs': runs, 'balls': balls, 'fours': fours, 'sixes': sixes,
                'dismissals': dismissals, 'strike_rate': sr, 'average': avg,
                'dot_pct': dot_pct, 'boundary_pct': boundary_pct}

    def bowler_stats(name):
        s = df[df['bowler'] == name]
        if len(s) == 0: return None
        balls = len(s)
        runs = int(s['total_runs'].sum())
        wickets = int(s['is_wicket'].sum())
        overs = round(balls / 6, 1)
        economy = round(runs / overs, 2) if overs > 0 else 0
        sr = round(balls / wickets, 1) if wickets > 0 else None
        dot_pct = round((s['total_runs'] == 0).sum() / balls * 100, 1)
        wicket_prob = round(wickets / balls * 100, 2)
        avg = round(runs / wickets, 2) if wickets > 0 else runs
        fours = int((s['batsman_runs'] == 4).sum())
        sixes = int((s['batsman_runs'] == 6).sum())
        return {'name': name, 'wickets': wickets, 'overs': overs, 'runs_conceded': runs,
                'economy': economy, 'strike_rate': sr, 'dot_pct': dot_pct,
                'wicket_prob': wicket_prob, 'average': avg, 'fours': fours, 'sixes': sixes}

    if mode == 'batter_batter':
        a = batter_stats(d['player1'])
        b = batter_stats(d['player2'])
        if not a or not b: return jsonify({'error': 'Player not found'})
        return jsonify({'mode': mode, 'a': a, 'b': b})

    elif mode == 'bowler_bowler':
        a = bowler_stats(d['player1'])
        b = bowler_stats(d['player2'])
        if not a or not b: return jsonify({'error': 'Player not found'})
        return jsonify({'mode': mode, 'a': a, 'b': b})

    elif mode == 'batter_bowler':
        a = batter_stats(d['player1'])
        b = bowler_stats(d['player2'])
        if not a or not b: return jsonify({'error': 'Player not found'})
        head2head = df[(df['batter'] == d['player1']) & (df['bowler'] == d['player2'])]
        h2h = {}
        if len(head2head) > 0:
            h2h = {
                'balls': int(len(head2head)),
                'runs': int(head2head['batsman_runs'].sum()),
                'dismissals': int(head2head['is_wicket'].sum()),
                'strike_rate': round(head2head['batsman_runs'].sum() / len(head2head) * 100, 1),
                'fours': int((head2head['batsman_runs'] == 4).sum()),
                'sixes': int((head2head['batsman_runs'] == 6).sum())
            }
        return jsonify({'mode': mode, 'a': a, 'b': b, 'head2head': h2h})

    return jsonify({'error': 'Invalid mode'})

# ── Pre-compute aggregates at startup for fast auction profile ──
_batter_agg = df.groupby('batter').agg(
    runs=('batsman_runs', 'sum'),
    balls_faced=('extra_runs', lambda x: (x == 0).sum()),
    fours=('batsman_runs', lambda x: (x == 4).sum()),
    sixes=('batsman_runs', lambda x: (x == 6).sum()),
    total_balls=('batsman_runs', 'count')
).reset_index()
_dismissals = df[df['is_wicket'] == 1].groupby('player_dismissed').size().reset_index(name='dismissals')
_batter_agg = _batter_agg.merge(_dismissals, left_on='batter', right_on='player_dismissed', how='left')
_batter_agg['dismissals'] = _batter_agg['dismissals'].fillna(0).astype(int)
_batter_agg['strike_rate'] = (_batter_agg['runs'] / _batter_agg['balls_faced'] * 100).round(2)
_batter_agg['average'] = (_batter_agg['runs'] / _batter_agg['dismissals'].replace(0, float('nan'))).fillna(_batter_agg['runs']).round(2)

_bowler_agg = df.groupby('bowler').agg(
    total_runs=('total_runs', 'sum'),
    wickets=('is_wicket', 'sum'),
    balls=('total_runs', 'count'),
    fours_conceded=('batsman_runs', lambda x: (x == 4).sum()),
    sixes_conceded=('batsman_runs', lambda x: (x == 6).sum())
).reset_index()
_bowler_agg['overs'] = (_bowler_agg['balls'] / 6).round(1)
_bowler_agg['economy'] = (_bowler_agg['total_runs'] / _bowler_agg['overs'].replace(0, float('nan'))).round(2)
_bowler_agg['strike_rate'] = (_bowler_agg['balls'] / _bowler_agg['wickets'].replace(0, float('nan'))).round(1)

@lru_cache(maxsize=256)
def _auction_profile_cached(name):
    is_batter = name in df['batter'].values
    is_bowler = name in df['bowler'].values
    if not is_batter and not is_bowler:
        return {'error': 'Player not found'}
    result = {'name': name, 'role': []}
    if is_batter:
        result['role'].append('batter')
        p = _batter_agg[_batter_agg['batter'] == name].iloc[0]
        runs, balls, fours, sixes, dismissals = int(p['runs']), int(p['balls_faced']), int(p['fours']), int(p['sixes']), int(p['dismissals'])
        sr, avg = float(p['strike_rate']), float(p['average'])
        result['batter_stats'] = {'runs': runs, 'balls': balls, 'fours': fours, 'sixes': sixes,
                                   'dismissals': dismissals, 'strike_rate': sr, 'average': avg}
        cands = _batter_agg[(_batter_agg['batter'] != name) & (_batter_agg['balls_faced'] >= 30)].copy()
        cands['_score'] = (cands['strike_rate'] - sr).abs() + (cands['runs'] - runs).abs() / 100
        cands = cands.nsmallest(10, '_score')
        result['similar_batters'] = cands[['batter','runs','balls_faced','fours','sixes','strike_rate','average']]\
            .rename(columns={'batter':'name','balls_faced':'balls'}).to_dict(orient='records')
        h2h = df[df['batter'] == name].groupby('bowler').agg(
            balls=('batsman_runs', 'count'), runs_conceded=('batsman_runs', 'sum'),
            dismissals=('is_wicket', 'sum'),
            fours=('batsman_runs', lambda x: (x == 4).sum()),
            sixes=('batsman_runs', lambda x: (x == 6).sum())
        ).reset_index()
        h2h = h2h[h2h['balls'] >= 6].copy()
        h2h['batter_sr'] = (h2h['runs_conceded'] / h2h['balls'] * 100).round(1)
        h2h = h2h.sort_values(['dismissals','batter_sr'], ascending=[False, True]).head(10)
        result['tough_bowlers'] = h2h.rename(columns={'bowler':'name'}).to_dict(orient='records')
    if is_bowler:
        result['role'].append('bowler')
        p = _bowler_agg[_bowler_agg['bowler'] == name].iloc[0]
        balls, overs, runs, wickets = int(p['balls']), float(p['overs']), int(p['total_runs']), int(p['wickets'])
        economy = float(p['economy']) if pd.notna(p['economy']) else 0
        sr_b = float(p['strike_rate']) if pd.notna(p['strike_rate']) else None
        result['bowler_stats'] = {'balls': balls, 'overs': overs, 'runs_conceded': runs,
                                   'wickets': wickets, 'economy': economy, 'strike_rate': sr_b,
                                   'fours_conceded': int(p['fours_conceded']), 'sixes_conceded': int(p['sixes_conceded'])}
        cands = _bowler_agg[(_bowler_agg['bowler'] != name) & (_bowler_agg['balls'] >= 30)].copy()
        cands['_score'] = (cands['economy'] - economy).abs() + (cands['wickets'] - wickets).abs() / 10
        cands = cands.nsmallest(10, '_score')
        result['similar_bowlers'] = cands[['bowler','wickets','overs','total_runs','economy','strike_rate','fours_conceded','sixes_conceded']]\
            .rename(columns={'bowler':'name','total_runs':'runs_conceded'}).to_dict(orient='records')
        h2h = df[df['bowler'] == name].groupby('batter').agg(
            balls=('batsman_runs', 'count'), runs=('batsman_runs', 'sum'),
            dismissals=('is_wicket', 'sum'),
            fours=('batsman_runs', lambda x: (x == 4).sum()),
            sixes=('batsman_runs', lambda x: (x == 6).sum())
        ).reset_index()
        h2h = h2h[h2h['balls'] >= 6].copy()
        h2h['strike_rate'] = (h2h['runs'] / h2h['balls'] * 100).round(1)
        h2h = h2h.sort_values(['runs','strike_rate'], ascending=[False, False]).head(10)
        result['tough_batters'] = h2h.rename(columns={'batter':'name'}).to_dict(orient='records')
    return result

@app.route('/auction/profile', methods=['POST'])
def auction_profile():
    return jsonify(_auction_profile_cached(request.json['name']))

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
