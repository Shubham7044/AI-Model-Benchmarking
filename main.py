import os
import matplotlib.pyplot as plt
import pandas as pd

from src.data_preprocessing import preprocess_data
from src.sklearn_models import run_sklearn_models
from src.tensorflow_model import run_tensorflow_model
from src.pytorch_model import run_pytorch_model


DATASETS = {
    "heart_disease": {
        "path": "data/heart_disease.csv",
        "target": "target"
    },
    "churn": {
        "path": "data/churn.csv",
        "target": "Churn"
    },
    "fraud": {
        "path": "data/fraud.csv",
        "target": "Class"
    }
}


def run_benchmark(name, path, target):

    print(f"\n===== Running Benchmark: {name} =====\n")

    X_train, X_test, y_train, y_test = preprocess_data(path, target)

    results = {}

    results.update(
        run_sklearn_models(X_train, X_test, y_train, y_test)
    )

    results["TensorFlow"] = run_tensorflow_model(
        X_train, X_test, y_train, y_test
    )

    results["PyTorch"] = run_pytorch_model(
        X_train, X_test, y_train, y_test
    )

    # 🔥 Mini AutoML Leaderboard
    leaderboard = sorted(
        results.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("\n🏆 Model Leaderboard\n")

    for rank, (model, score) in enumerate(leaderboard, 1):

        print(f"{rank}. {model}: {score:.4f}")

    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame(
        leaderboard,
        columns=["Model", "Accuracy"]
    )

    df.to_csv(f"results/{name}_results.csv", index=False)

    plt.figure()

    models = [x[0] for x in leaderboard]
    scores = [x[1] for x in leaderboard]

    plt.bar(models, scores)

    plt.title(f"{name} Benchmark")

    plt.ylabel("Accuracy")

    plt.xticks(rotation=30)

    plt.tight_layout()

    plt.savefig(f"results/{name}_chart.png")

    plt.close()


def main():

    for dataset, config in DATASETS.items():

        run_benchmark(
            dataset,
            config["path"],
            config["target"]
        )


if __name__ == "__main__":
    main()