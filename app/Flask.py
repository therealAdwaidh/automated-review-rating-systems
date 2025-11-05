from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------------
# Initialize Flask app
# -------------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------------
# Load models and tokenizers
# -------------------------------------------------------
MODEL_A_PATH = "models/A/Model_A.h5"
TOKENIZER_A_PATH = "models/A/tokenizer.pkl"

MODEL_B_PATH = "models/B/Model_B.h5"
TOKENIZER_B_PATH = "models/B/tokenizer_model_b.pkl"

# Load both models and tokenizers
model_a = load_model(MODEL_A_PATH)
with open(TOKENIZER_A_PATH, "rb") as f:
    tokenizer_a = pickle.load(f)

model_b = load_model(MODEL_B_PATH)
with open(TOKENIZER_B_PATH, "rb") as f:
    tokenizer_b = pickle.load(f)

# -------------------------------------------------------
# Helper function for predictions
# -------------------------------------------------------
def predict_rating(review_text, model, tokenizer, max_len=100):
    seq = tokenizer.texts_to_sequences([review_text])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    preds = model.predict(padded)
    rating = int(np.argmax(preds) + 1)  # Convert to 1–5 scale
    confidence = float(np.max(preds))
    return rating, confidence

# -------------------------------------------------------
# Flask Routes
# -------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        review = request.form["review"]
        if review.strip():
            rating_a, conf_a = predict_rating(review, model_a, tokenizer_a)
            rating_b, conf_b = predict_rating(review, model_b, tokenizer_b)

            return render_template(
                "index.html",
                review=review,
                rating_a=rating_a,
                conf_a=round(conf_a * 100, 2),
                rating_b=rating_b,
                conf_b=round(conf_b * 100, 2),
                result=True,
            )
    return render_template("index.html", result=False)

# -------------------------------------------------------
# Run app
# -------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
