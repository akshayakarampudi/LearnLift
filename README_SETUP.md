# LearnLift — improved local project

## Run on Windows (PowerShell)

1. Extract this ZIP.
2. Open the `LearnLift` folder in PyCharm.
3. In the terminal, make sure the terminal path ends with `\LearnLift`.
4. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

5. Start the Flask app:

   ```powershell
   python app.py
   ```

6. Open http://127.0.0.1:5000 in your browser. Keep the terminal open while using the website. Press Ctrl+C to stop it.

## What was improved
- Added a responsive purple/white theme that restyles the existing pages without replacing the model or route logic.
- Added `requirements.txt` so dependencies can be installed together.
- Kept the existing OULAD data, saved classifier/preprocessor, clustering model, plots, templates, and training scripts.
- Excluded the local `.venv`, IDE settings, and Git internals from the delivery ZIP to make it smaller and easier to extract.

## Important
The prediction results depend on the existing saved model and its training data. This project is an educational aid; predictions should not be used as the sole basis for decisions about students.
