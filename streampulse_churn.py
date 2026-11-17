import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay

np.random.seed(42)

n = 2000
plan = np.random.choice(["Basic", "Standard", "Premium"], n, p=[0.4, 0.4, 0.2])
fee_map = {"Basic": 7.99, "Standard": 12.99, "Premium": 17.99}
df = pd.DataFrame({
    "age": np.random.randint(18, 70, n).astype(float),
    "tenure_months": np.random.randint(1, 60, n),
    "plan": plan,
    "monthly_fee": [fee_map[p] for p in plan],
    "contract": np.random.choice(["Monthly", "Annual"], n, p=[0.6, 0.4]),
    "payment_method": np.random.choice(["Card", "PayPal", "Bank"], n, p=[0.5, 0.3, 0.2]).astype(object),
    "watch_hours_per_week": np.clip(np.random.normal(10, 5, n), 0, None),
    "support_tickets": np.random.poisson(1.2, n),
    "days_since_last_login": np.random.exponential(10, n).round()
})

logit = (
    -0.5
    + 0.05 * df["days_since_last_login"]
    - 0.03 * df["tenure_months"]
    - 0.08 * df["watch_hours_per_week"]
    + 0.4 * df["support_tickets"]
    + 0.6 * (df["contract"] == "Monthly")
    + 0.08 * (df["monthly_fee"] - 12.99)
)
prob = 1 / (1 + np.exp(-logit))
df["churn"] = np.random.binomial(1, prob)

for col in ["age", "watch_hours_per_week", "payment_method"]:
    idx = np.random.choice(n, int(n * 0.03), replace=False)
    df.loc[idx, col] = np.nan

print("Перші рядки даних:")
print(df.head())
print("\nПропущені значення:")
print(df.isna().sum())
print("\nРозподіл цільової змінної (churn):")
print(df["churn"].value_counts())
print(df["churn"].value_counts(normalize=True).round(3))

X = df.drop(columns="churn")
y = df["churn"]

num_cols = ["age", "tenure_months", "monthly_fee", "watch_hours_per_week", "support_tickets", "days_since_last_login"]
cat_cols = ["plan", "contract", "payment_method"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
}

pipelines = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []

for name, clf in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", clf)])
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    pipelines[name] = pipe
    rows.append({
        "Модель": name,
        "CV ROC-AUC": round(cv_auc.mean(), 4),
        "Accuracy": round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred), 4),
        "Recall": round(recall_score(y_test, pred), 4),
        "F1": round(f1_score(y_test, pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, proba), 4)
    })

results = pd.DataFrame(rows)
print("\nПорівняння моделей на тестовій вибірці:")
print(results.to_string(index=False))

best_name = results.sort_values("ROC-AUC", ascending=False).iloc[0]["Модель"]
best = pipelines[best_name]
print("\nНайкраща модель:", best_name)
print(classification_report(y_test, best.predict(X_test), target_names=["Залишається", "Піде (churn)"]))

fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for name, pipe in pipelines.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=axs[0], name=name)
axs[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
axs[0].set_title("ROC-криві")
axs[0].grid(True)
ConfusionMatrixDisplay.from_estimator(best, X_test, y_test, ax=axs[1], cmap="Blues", display_labels=["Залишається", "Піде"], colorbar=False)
axs[1].set_title(f"Матриця помилок: {best_name}")
plt.tight_layout()
plt.show()

forest = pipelines["Random Forest"]
names = forest.named_steps["preprocessor"].get_feature_names_out()
importances = pd.Series(forest.named_steps["model"].feature_importances_, index=names).sort_values()
plt.figure(figsize=(9, 6))
importances.tail(10).plot(kind="barh", color="teal")
plt.xlabel("Важливість ознаки")
plt.title("Топ-10 ознак за важливістю (Random Forest)")
plt.tight_layout()
plt.show()

new_users = pd.DataFrame({
    "age": [25, 45, 33],
    "tenure_months": [2, 40, 8],
    "plan": ["Premium", "Basic", "Standard"],
    "monthly_fee": [17.99, 7.99, 12.99],
    "contract": ["Monthly", "Annual", "Monthly"],
    "payment_method": ["Card", "Bank", "PayPal"],
    "watch_hours_per_week": [1.0, 18.0, 4.0],
    "support_tickets": [4, 0, 2],
    "days_since_last_login": [30, 1, 15]
})
new_proba = best.predict_proba(new_users)[:, 1]

print("Прогноз для нових користувачів:")
for i, p in enumerate(new_proba):
    action = "запропонувати знижку або спеціальну пропозицію" if p >= 0.5 else "додаткових дій не потрібно"
    print(f"Користувач {i + 1}: ймовірність відтоку {p:.2f} -> {action}")
