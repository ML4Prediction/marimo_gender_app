# Webcam gender classifier

A marimo app that reads your webcam, finds faces, and writes the predicted gender on top of each face.
The model (a neural network without hidden layers) is trained in R on [FairFace](https://github.com/joojs/fairface)
by [`../app_model/train_model.qmd`](../app_model/train_model.qmd) and saved here as `gender_model.keras`.

## Run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
cd app
uv run marimo edit app.py
```

On macOS, allow camera access for your terminal when asked.

## Retrain

Run `../app_model/train_model.qmd` in RStudio (needs the R packages `keras3`, `arrow` and `jpeg`, and ~4 GB of RAM).
It downloads FairFace (~310 MB) to `app_model/data/` and overwrites `gender_model.keras`.
If you change the 32 x 32 input size, change it in `app.py` too.

## Exercise

Without hidden layers (logistic regression on the pixels) the model reaches ~74% validation accuracy.
Add hidden layers in the `.qmd`, retrain, and see if the accuracy and the live predictions improve.
