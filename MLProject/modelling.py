import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow


def run_ci_modelling():
    print("Membaca dataset untuk CI Pipeline...")
    df = pd.read_csv("mobile_price_preprocessing/clean_mobile_price.csv")

    X = df.drop("price_range", axis=1)
    y = df["price_range"]

    print("Mengaktifkan MLflow Autolog...")
    mlflow.autolog()

    with mlflow.start_run(run_name="ci_training_run"):
        print("Melakukan Re-Training Model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        print("Selesai! Model berhasil dilatih dan dilog ke MLflow.")


if __name__ == "__main__":
    run_ci_modelling()
