import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle

def train():
    df = pd.read_csv('deliveries.csv')

    # Aggregate: runs per batter-bowler matchup per match
    matchup = df.groupby(['match_id', 'batter', 'bowler']).agg(
        balls=('ball', 'count'),
        runs=('batsman_runs', 'sum'),
        dismissed=('is_wicket', 'sum')
    ).reset_index()

    encoders = {}
    for col in ['batter', 'bowler']:
        le = LabelEncoder()
        matchup[col] = le.fit_transform(matchup[col].astype(str))
        encoders[col] = le

    X = matchup[['batter', 'bowler', 'balls', 'dismissed']]
    y = matchup['runs']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

    y_pred = model.predict(X_test)
    print("\n--- Batsman vs Bowler Matchup ---")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"R2:  {r2_score(y_test, y_pred):.4f}")

    with open('matchup_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('matchup_encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    print("Models saved: matchup_model.pkl, matchup_encoders.pkl")

if __name__ == '__main__':
    train()
