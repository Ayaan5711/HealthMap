from flask import Flask, render_template, request, jsonify
import json

app = Flask(__name__)

# Load side effects data from JSON file
with open('side_effects.json', 'r') as file:
    side_effects_data = json.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    side_effects = None
    if request.method == 'POST':
        drug_name = request.form.get('drug_name')
        side_effects = side_effects_data.get(drug_name, "No data available for this drug.")
    return render_template('drug-response-index.html', side_effects=side_effects[:25])

if __name__ == '__main__':
    app.run(debug=True)
