
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# =====================================
# 1. LOAD DATASET
# =====================================

print("Loading dataset...")

train_data = pd.read_csv("sign_mnist_train.csv")
test_data = pd.read_csv("sign_mnist_test.csv")

print("Training dataset shape:", train_data.shape)
print("Testing dataset shape:", test_data.shape)

# =====================================
# 2. PREPARE TRAINING DATA
# =====================================

X = train_data.drop("label", axis=1).values
y = train_data["label"].values

X = X.astype("float32") / 255.0

X = X.reshape(-1, 28, 28, 1)

# =====================================
# 3. PREPARE TEST DATA
# =====================================

# Check whether test dataset contains labels
if "label" in test_data.columns:

    X_test = test_data.drop("label", axis=1).values
    y_test = test_data["label"].values

    X_test = X_test.astype("float32") / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1)

    has_test_labels = True

else:

    print("Warning: Test dataset has no labels.")
    print("Test accuracy cannot be calculated.")

    X_test = test_data.values.astype("float32") / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1)

    has_test_labels = False

# =====================================
# 4. SPLIT TRAINING AND VALIDATION
# =====================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))

# =====================================
# 5. DATA AUGMENTATION
# =====================================

augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomRotation(0.05),

    tf.keras.layers.RandomTranslation(
        height_factor=0.08,
        width_factor=0.08
    ),

    tf.keras.layers.RandomZoom(0.08)

])

# =====================================
# 6. BUILD IMPROVED CNN
# =====================================

print("Building improved CNN model...")

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(28, 28, 1)),

    augmentation,

    # First convolution block
    tf.keras.layers.Conv2D(
        32, (3, 3),
        padding="same",
        use_bias=False
    ),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Second convolution block
    tf.keras.layers.Conv2D(
        64, (3, 3),
        padding="same",
        use_bias=False
    ),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Third convolution block
    tf.keras.layers.Conv2D(
        128, (3, 3),
        padding="same",
        use_bias=False
    ),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),

    # Classification layers
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(25, activation="softmax")

])

# =====================================
# 7. COMPILE MODEL
# =====================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

# =====================================
# 8. CALLBACKS
# =====================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001
    )

]

# =====================================
# 9. TRAIN MODEL
# =====================================

print("Training started...")

history = model.fit(

    X_train,
    y_train,

    epochs=30,

    batch_size=128,

    validation_data=(X_val, y_val),

    callbacks=callbacks

)

# =====================================
# 10. EVALUATE MODEL
# =====================================

print("\nEvaluating model...")

# Validation accuracy
val_loss, val_accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0
)

print(
    "Validation Accuracy:",
    round(val_accuracy * 100, 2),
    "%"
)

# Test accuracy only when labels exist
if has_test_labels:

    print("Test images:", len(X_test))
    print("Test labels:", len(y_test))

    if len(X_test) == len(y_test):

        predictions = model.predict(X_test)

        predicted_labels = np.argmax(
            predictions,
            axis=1
        )

        accuracy = accuracy_score(
            y_test,
            predicted_labels
        )

        print(
            "\nTest Accuracy:",
            round(accuracy * 100, 2),
            "%"
        )

        print("\nClassification Report:")

        print(
            classification_report(
                y_test,
                predicted_labels
            )
        )

    else:

        print("Error: Test image and label counts do not match.")
        print("Test accuracy was not calculated.")

# =====================================
# 11. SAVE MODEL
# =====================================

model.save("sign_language_model.keras")

print("\nModel saved successfully!")
print("You can now run predict.py")
