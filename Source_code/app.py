from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    with open('model.pkl', 'rb') as le:
        model = pickle.load(le)
except FileNotFoundError:
    model = None
    print("Error: Model file not found. Make sure 'model.pkl' is in the same directory.")
except Exception as e:
    model = None
    print(f"Error loading model: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return render_template('index.html', prediction_text="Error: Model could not be loaded.")

        # Collect form inputs
        gender = request.form['gender']
        country = request.form['country']
        occupation = request.form['occupation']
        self_employed = request.form['self_employed'] if request.form['self_employed'] else None
        family_history = request.form['family_history']
        days_indoors = request.form['days_indoors']
        growing_stress = request.form['growing_stress']
        changes_habits = request.form['changes_habits']
        mental_health_history = request.form['mental_health_history']
        mood_swings = request.form['mood_swings']
        coping_struggles = request.form['coping_struggles'] if request.form['coping_struggles'] else None
        work_interest = request.form['work_interest'] if request.form['work_interest'] else None
        social_weakness = request.form['social_weakness'] if request.form['social_weakness'] else None
        mental_health_interview = request.form['mental_health_interview'] if request.form['mental_health_interview'] else None
        care_options = request.form['care_options']

        # Required field validation
        required_fields = [
            'gender', 'country', 'occupation', 'family_history', 'days_indoors',
            'growing_stress', 'changes_habits', 'mental_health_history', 'mood_swings', 'care_options'
        ]
        for field in required_fields:
            if not request.form.get(field):
                return render_template('index.html', prediction_text=f"Error: {field} is required.")

        # Convert to integers
        try:
            gender = int(gender)
            country = int(country)
            occupation = int(occupation)
            family_history = int(family_history)
            days_indoors = int(days_indoors)
            growing_stress = int(growing_stress)
            changes_habits = int(changes_habits)
            mental_health_history = int(mental_health_history)
            mood_swings = int(mood_swings)
            care_options = int(care_options)
            self_employed = int(self_employed) if self_employed else None
            coping_struggles = int(coping_struggles) if coping_struggles else None
            work_interest = int(work_interest) if work_interest else None
            social_weakness = int(social_weakness) if social_weakness else None
            mental_health_interview = int(mental_health_interview) if mental_health_interview else None
        except ValueError:
            return render_template('index.html', prediction_text="Error: Invalid input types.")

        # Input array
        input_features = np.array([gender, country, occupation, self_employed, family_history,
                                   days_indoors, growing_stress, changes_habits, mental_health_history, 
                                   mood_swings, coping_struggles, work_interest, social_weakness, 
                                   mental_health_interview, care_options]).reshape(1, -1)

        # Prediction
        prediction = model.predict(input_features)

        # Friendly output
        if prediction[0] == 0:
            output = (
                "Prediction: Treatment Need\n"
                "Based on your responses, our system suggests you may benefit from seeking mental health support. "
                "Please consider consulting a mental health professional for further guidance."
            )
        else:
            output = (
                "Prediction: No Treatment Needed\n"
                "Your responses do not currently indicate signs that require professional mental health intervention. "
                "However, continue to monitor your mental well-being and seek support if needed."
            )

        return render_template('index.html', prediction_text=output)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
