import tensorflow as tf
from sklearn.metrics import accuracy_score


def run_tensorflow_model(X_train, X_test, y_train, y_test):

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    preds = model.predict(X_test)
    preds = (preds > 0.5).astype(int)

    acc = accuracy_score(y_test, preds)

    print(f"TensorFlow Accuracy: {acc:.4f}")

    return acc