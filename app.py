
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load Dataset
df = pd.read_csv("student_data.csv")
df["Result"] = df["Result"].map({"Fail": 0, "Pass": 1})

X = df[["Study_Hours", "Attendance", "Internal_Marks", "Assignment_Marks"]]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    study_hours = float(request.form["study_hours"])
    attendance = float(request.form["attendance"])
    internal = float(request.form["internal"])
    assignment = float(request.form["assignment"])

    new_data = [[study_hours, attendance, internal, assignment]]
    prediction = model.predict(new_data)

    result = "PASS ✅" if prediction[0] == 1 else "FAIL ❌"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)