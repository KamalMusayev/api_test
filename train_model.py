import pickle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    iris = load_iris()

    X = iris.data
    y = iris.target
    class_names = iris.target_names.tolist()
    feature_names = iris.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=200))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    artifact = {
        "model": model,
        "class_names": class_names,
        "feature_names": feature_names,
        "accuracy": accuracy
    }

    with open("model.pkl", "wb") as file:
        pickle.dump(artifact, file)

    print("Model trained successfully.")
    print(f"Accuracy: {accuracy:.4f}")
    print("Saved as model.pkl")
    print("Features:")
    for name in feature_names:
        print(f"- {name}")


if __name__ == "__main__":
    main()
