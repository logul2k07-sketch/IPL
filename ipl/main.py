print("=" * 50)
print("IPL Ball-by-Ball ML Pipeline")
print("=" * 50)

print("\n[1/5] Wicket Prediction...")
from wicket_prediction import train as train_wicket
train_wicket()

print("\n[2/5] Runs Per Ball Prediction...")
from runs_prediction import train as train_runs
train_runs()

print("\n[3/5] Total Score Predictor...")
from score_predictor import train as train_score
train_score()

print("\n[4/5] Bowler Performance Analyzer...")
from bowler_analyzer import analyze
analyze()

print("\n[5/5] Batsman vs Bowler Matchup...")
from matchup_model import train as train_matchup
train_matchup()

print("\nAll models trained and saved!")
