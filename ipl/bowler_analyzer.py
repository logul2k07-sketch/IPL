import pandas as pd
import pickle

def analyze(top_n=20):
    df = pd.read_csv('deliveries.csv')

    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)

    bowler_stats = df.groupby('bowler').agg(
        balls=('ball', 'count'),
        runs_conceded=('total_runs', 'sum'),
        wickets=('is_wicket', 'sum')
    ).reset_index()

    bowler_stats['overs'] = bowler_stats['balls'] / 6
    bowler_stats['economy'] = bowler_stats['runs_conceded'] / bowler_stats['overs']
    bowler_stats['strike_rate'] = bowler_stats['balls'] / bowler_stats['wickets'].replace(0, float('nan'))
    bowler_stats['wicket_prob'] = bowler_stats['wickets'] / bowler_stats['balls']

    # Filter bowlers with at least 60 balls
    bowler_stats = bowler_stats[bowler_stats['balls'] >= 60].sort_values('wickets', ascending=False)

    print("\n--- Top Bowlers by Wickets ---")
    print(bowler_stats[['bowler', 'wickets', 'economy', 'strike_rate', 'wicket_prob']].head(top_n).to_string(index=False))

    bowler_stats.to_csv('bowler_analysis.csv', index=False)
    print("\nSaved: bowler_analysis.csv")
    return bowler_stats

if __name__ == '__main__':
    analyze()
