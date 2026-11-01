# Fruit Detection System — Streamlit

## Files
- `app.py`: Streamlit image-upload and YOLO detection interface.
- `best.pt`: trained Ultralytics YOLO checkpoint.
- `requirements.txt`: Python dependencies.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Push these files to a GitHub repository. In Streamlit Community Cloud,
create an app from that repository and set the main file to `app.py`.

The checkpoint was supplied as `best_corrected.pt.zip` and extracted into
the Ultralytics checkpoint file `best.pt`.
