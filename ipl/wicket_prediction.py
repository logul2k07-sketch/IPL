from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pickle
from data_preprocessing import load_and_preprocess, FEATURES

def train():
    df, _ = load_and_preprocess()

    X = df[FEATURES]
    y = df['is_wicket']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
                          use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n--- Wicket Prediction ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    with open('wicket_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved: wicket_model.pkl")

if __name__ == '__main__':
    train()
