from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
from data_preprocessing import load_and_preprocess, FEATURES

def train():
    df, _ = load_and_preprocess()

    X = df[FEATURES]
    y = df['batsman_runs'].clip(0, 6)  # 0-6 runs per ball

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

    y_pred = model.predict(X_test)
    print("\n--- Runs Per Ball Prediction ---")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    with open('runs_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved: runs_model.pkl")

if __name__ == '__main__':
    train()
