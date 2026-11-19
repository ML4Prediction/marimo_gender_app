import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import os

    os.environ["KERAS_BACKEND"] = "jax"

    import base64

    import anywidget
    import cv2
    import keras
    import marimo as mo
    import traitlets
    return anywidget, base64, cv2, keras, mo, traitlets


@app.cell
def _(keras):
    # Architecture + weights, trained in R: ../app_model/train_model.qmd
    model = keras.saving.load_model("gender_model.keras")
    model.summary()
    return (model,)


@app.cell
def _(anywidget, traitlets):
    class Video(anywidget.AnyWidget):
        """Shows JPEG frames in a single <img> that is updated in place (no flicker)."""

        _esm = """
        export default { render({ model, el }) {
          const img = document.createElement("img");
          const show = () => { img.src = "data:image/jpeg;base64," + model.get("frame"); };
          model.on("change:frame", show);
          show();
          el.appendChild(img);
        }};
        """
        frame = traitlets.Unicode().tag(sync=True)
    return (Video,)


@app.cell
def _(Video, cv2, mo):
    camera = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    video = Video()
    refresh = mo.ui.refresh(options=["0.1s", "0.2s", "0.5s", "1s"], default_interval="0.5s")
    mo.vstack([refresh, video])
    return camera, face_detector, refresh, video


@app.cell
def _(base64, camera, cv2, face_detector, mo, model, refresh, video):
    refresh
    _ok, _frame = camera.read()
    mo.stop(not _ok, mo.md("No camera image. Check that your camera is free and that the terminal has camera permission."))

    _gray = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)
    for _x, _y, _w, _h in face_detector.detectMultiScale(_gray, 1.1, 5):
        _y += int(0.09 * _h)  # FairFace crops sit 9% lower than Haar boxes; move down to match
        _face = cv2.resize(_frame[_y : _y + _h, _x : _x + _w], (32, 32), interpolation=cv2.INTER_AREA)
        _p = model.predict(_face[None, :, :, ::-1] / 255, verbose=0)[0, 0]  # BGR -> RGB, like in R
        _label = f"female {_p:.0%}" if _p > 0.5 else f"male {1 - _p:.0%}"
        cv2.rectangle(_frame, (_x, _y), (_x + _w, _y + _h), (0, 255, 0), 2)
        cv2.putText(_frame, _label, (_x, _y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    video.frame = base64.b64encode(cv2.imencode(".jpg", _frame)[1]).decode()
    return


if __name__ == "__main__":
    app.run()
