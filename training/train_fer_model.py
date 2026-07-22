"""
Trains a CNN on the FER-2013 image-folder dataset and saves the result to
app/ml_models/fer2013_model.h5 - the exact path app/services/fer_service.py
already expects.

HOW TO RUN THIS (from the `backend/` folder, with venv activated):

    pip install scikit-learn
    python training/train_fer_model.py

Expected folder layout (see training/README.md for full setup instructions):

    backend/
      training/
        train_fer_model.py   <- this file
        dataset/
          train/
            angry/*.jpg
            disgust/*.jpg
            fear/*.jpg
            happy/*.jpg
            neutral/*.jpg
            sad/*.jpg
            surprise/*.jpg
          test/
            angry/*.jpg
            ... (same 7 folders)

This will take a while - CPU training on ~35,000 images is realistically
1-3 hours depending on your machine (early stopping usually cuts it short
before all epochs run). A GPU (e.g. via Google Colab) finishes in minutes.
"""
import pathlib

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

BACKEND_DIR = SCRIPT_DIR.parent
MODEL_OUTPUT_PATH = BACKEND_DIR / "app" / "ml_models" / "fer2013_model.h5"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 40  # early stopping will likely stop well before this
VALIDATION_SPLIT = 0.1
SEED = 42

# IMPORTANT: this exact order must match EMOTION_LABELS in
# app/services/fer_service.py so the model's output indices line up with
# what the API expects. Do not reorder this list.
EMOTION_ORDER = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def check_dataset_exists():
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Couldn't find the dataset at {DATASET_DIR}.\n"
            f"Extract your Kaggle archive so that '{TRAIN_DIR}' and "
            f"'{TEST_DIR}' both exist, each containing the 7 emotion "
            f"subfolders (angry, disgust, fear, happy, neutral, sad, surprise)."
        )
    for emotion in EMOTION_ORDER:
        if not (TRAIN_DIR / emotion).exists():
            raise FileNotFoundError(f"Missing expected folder: {TRAIN_DIR / emotion}")


def build_generators():
    # Training data gets light augmentation to help the model generalize
    # beyond the exact pixels it has seen. Validation/test data is NOT
    # augmented - we want to measure real performance, not performance on
    # artificially altered images.
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=VALIDATION_SPLIT,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    common_args = dict(
        target_size=IMG_SIZE,
        color_mode="grayscale",
        class_mode="categorical",
        classes=EMOTION_ORDER,  # forces folder->index mapping to match EMOTION_ORDER
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    train_gen = train_datagen.flow_from_directory(TRAIN_DIR, subset="training", shuffle=True, **common_args)
    val_gen = train_datagen.flow_from_directory(TRAIN_DIR, subset="validation", shuffle=False, **common_args)
    test_gen = test_datagen.flow_from_directory(TEST_DIR, shuffle=False, **common_args)

    return train_gen, val_gen, test_gen


def build_model():
    model = models.Sequential([
        layers.Input(shape=(48, 48, 1)),

        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(256, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(len(EMOTION_ORDER), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def compute_weights(train_gen):
    # `disgust` has ~6x fewer training images than `happy` in this dataset.
    # Without class weighting, the model would learn to almost never
    # predict the underrepresented classes because doing so barely hurts
    # the loss. This reweights each class inversely to its frequency.
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_gen.classes),
        y=train_gen.classes,
    )
    return dict(enumerate(weights))


def main():
    check_dataset_exists()
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    train_gen, val_gen, test_gen = build_generators()
    print(f"Train samples: {train_gen.samples} | Validation: {val_gen.samples} | Test: {test_gen.samples}")

    class_weights = compute_weights(train_gen)
    print("Class weights (to correct for imbalance):", dict(zip(EMOTION_ORDER, [round(class_weights[i], 2) for i in range(7)])))

    model = build_model()
    model.summary()

    callbacks_list = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        callbacks.ModelCheckpoint(str(MODEL_OUTPUT_PATH), monitor="val_accuracy", save_best_only=True),
    ]

    print("\nStarting training - this will take a while...\n")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks_list,
    )

    print("\nEvaluating on held-out test set...")
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    # ModelCheckpoint already saved the best epoch, but save again to be safe
    # in case training ended without validation accuracy ever improving.
    model.save(str(MODEL_OUTPUT_PATH))
    print(f"\nModel saved to: {MODEL_OUTPUT_PATH}")
    print("Restart your backend (uvicorn) so it picks up the new model file.")


if __name__ == "__main__":
    main()
