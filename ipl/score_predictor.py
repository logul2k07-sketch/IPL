import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
from data_preprocessing import load_and_preprocess

def train():
    df, _ = load_and_preprocess()

    # Use only 1st innings data, aggregate per match at each ball
    innings1 = df[df['inning'] == 1].copy()

    # Final score per match
    final_scores = innings1.groupby('match_id')['total_runs'].sum().reset_index()
    final_scores.columns = ['match_id', 'final_score']

    innings1 = innings1.merge(final_scores, on='match_id')

    features = ['batting_team', 'bowling_team', 'over', 'ball', 'cum_runs', 'cum_wickets', 'balls_bowled']
    X = innings1[features]
    y = innings1['final_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

    y_pred = model.predict(X_test)
    print("\n--- Total Score Predictor ---")
    print(f"MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"R2:   {r2_score(y_test, y_pred):.4f}")

    with open('score_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved: score_model.pkl")

if __name__ == '__main__':
    train()
