from flask import Flask, render_template, request
import joblib

#Load Model and Scaler
model = joblib.load("medical_svc_model.lb")
scaler = joblib.load("medical_scaler.lb")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/history')
def history():
    return render_template('history.html')
@app.route('/project', methods=['GET', 'POST'])
def project():

    prediction = None

    if request.method == "POST":

        age = float(request.form["age"])
        bmi = float(request.form["bmi"])
        blood_pressure = float(request.form["blood_pressure"])
        glucose_level = float(request.form["glucose_level"])
        cholesterol = float(request.form["cholesterol"])
        heart_rate = float(request.form["heart_rate"])

        gender = int(request.form["gender"])
        smoking = int(request.form["smoking"])
        alcohol = int(request.form["alcohol"])
        physical_activity = int(request.form["physical_activity"])
        family_history = int(request.form["family_history"])

        data = [[
            age,
            gender,
            bmi,
            blood_pressure,
            glucose_level,
            cholesterol,
            heart_rate,
            smoking,
            alcohol,
            physical_activity,
            family_history
        ]]

        data = scaler.transform(data)
        pred = model.predict(data)

        disease_dict = {
            0: "Healthy",
            1: "Hypertension",
            2: "Pre-Diabetes",
            3: "Heart Disease",
            4: "Diabetes"
        }

        prediction = disease_dict[pred[0]]

    return render_template("project.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
