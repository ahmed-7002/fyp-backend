# Training the FER-2013 model (optional/advanced)

**This is no longer required** - `app/services/fer_service.py` now uses the
pre-trained `fer` Python package by default, so video/combined assessment
mode works out of the box with just `pip install -r requirements.txt`.

Only follow this guide if you specifically want to train your own custom
model instead (e.g. for a class project, to experiment with the CNN
architecture, or because you want full control over training data). If
that's not you, you can skip this whole folder.

## 1. Extract your Kaggle dataset here

Your `archive.zip` contains `train/` and `test/` folders, each with 7
subfolders (angry, disgust, fear, happy, neutral, sad, surprise) full of
48x48 grayscale `.jpg` images.

Extract it so the folders sit **inside a `dataset` folder right here**,
like this:

```
backend/
  training/
    train_fer_model.py
    dataset/              <- create this folder
      train/
        angry/
        disgust/
        fear/
        happy/
        neutral/
        sad/
        surprise/
      test/
        angry/
        disgust/
        fear/
        happy/
        neutral/
        sad/
        surprise/
```

Practically: create a folder named `dataset` inside `training/`, then take
the `train` and `test` folders straight out of your unzipped archive and
drop them both inside `dataset`.

## 2. Install the one extra dependency

From the `backend/` folder, with your venv already activated:

```bash
pip install scikit-learn
```

(Everything else the script needs - TensorFlow, NumPy - is already in
`requirements.txt` from the main setup.)

## 3. Run it

Still from the `backend/` folder:

```bash
python training/train_fer_model.py
```

You'll see progress print out per epoch (accuracy going up over time is
what you want to see). This can take **1-3 hours on a CPU-only laptop** -
that's normal for a dataset this size. If you have access to a GPU (e.g.
a free Google Colab notebook), it'll finish in a few minutes instead - the
script itself works unchanged either way, you'd just run it there and
copy the resulting `.h5` file back into `backend/app/ml_models/`.

## 4. What happens automatically

- The best-performing version of the model (based on validation accuracy)
  is saved to `backend/app/ml_models/fer2013_model.h5` during training.
- At the end, the script re-saves the final model there and prints a test
  accuracy score on the held-out `test/` folder.
- Class imbalance (the `disgust` folder has far fewer images than the
  others) is automatically corrected for via class weighting - you don't
  need to do anything about this yourself.

## After training finishes (custom model path only)

Since the default `fer_service.py` uses the bundled `fer` package, wiring
in your own trained model requires one manual edit: uncomment the
"custom model" block inside `_get_detector()` in
`app/services/fer_service.py` and comment out the `FER()` line above it.
See `backend/README.md` section 4 for exactly where.

```bash
uvicorn app.main:app --reload --port 8000
```

It will pick up the new model file automatically. Video and Combined
assessment modes will now work end-to-end.

## Troubleshooting

- **"Couldn't find the dataset" error** - double check the folder is
  exactly `training/dataset/train/...` and `training/dataset/test/...`,
  matching the layout above exactly (capitalization matters on some systems).
- **Training seems frozen** - it isn't; the first epoch is always slowest
  because TensorFlow is still warming up. Give it a few minutes before
  worrying.
- **Very low test accuracy (~15-20%)** - FER-2013 is a genuinely hard
  dataset even for published research (typical top models land around
  65-72% accuracy). Anything meaningfully above random guessing (1/7 ≈ 14%)
  means it's working; don't expect near-100% accuracy.
