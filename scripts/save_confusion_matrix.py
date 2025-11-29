import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

MODEL = "models/hate_speech_model_only.joblib"
VECT = "models/vectorizer_only.joblib"
DATA = "labeled.csv"   # change if your dataset filename is different
OUT = "docs/confusion_matrix.png"

if not os.path.exists(MODEL) or not os.path.exists(VECT):
    print("Model or vectorizer not found in 'models/'. Please put files in models/ and rerun.")
    raise SystemExit(0)

df = pd.read_csv(DATA)
vectorizer = joblib.load(VECT)
model = joblib.load(MODEL)

X = vectorizer.transform(df["tweet"])
y_true = df["class"].values
y_pred = model.predict(X)

labels = sorted(list(set(y_true) | set(y_pred)))
cm = confusion_matrix(y_true, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues", values_format='d')
plt.title("Confusion Matrix")
plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=150)
print("Saved", OUT)
