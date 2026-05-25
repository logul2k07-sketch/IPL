import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

def load_and_preprocess():
    df = pd.read_csv('deliveries.csv')

    # Fill nulls
    df['extras_type'] = df['extras_type'].fillna('none')
    df['player_dismissed'] = df['player_dismissed'].fillna('none')
    df['dismissal_kind'] = df['dismissal_kind'].fillna('none')
    df['fielder'] = df['fielder'].fillna('none')

    # Encode categorical columns
    encoders = {}
    cat_cols = ['batting_team', 'bowling_team', 'batter', 'bowler', 'extras_type', 'dismissal_kind']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Cumulative features per innings
    df = df.sort_values(['match_id', 'inning', 'over', 'ball'])
    df['cum_runs'] = df.groupby(['match_id', 'inning'])['total_runs'].cumsum() - df['total_runs']
    df['cum_wickets'] = df.groupby(['match_id', 'inning'])['is_wicket'].cumsum() - df['is_wicket']
    df['balls_bowled'] = df.groupby(['match_id', 'inning']).cumcount()

    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)

    return df, encoders

FEATURES = ['inning', 'batting_team', 'bowling_team', 'over', 'ball',
            'batter', 'bowler', 'cum_runs', 'cum_wickets', 'balls_bowled', 'extra_runs']
