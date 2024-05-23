from model_gui_backend import StartupSuccessPredModel
from dataset import load_dataset
from flask import Flask, request, render_template

startup_success_pred_model = StartupSuccessPredModel(load_dataset())
startup_success_pred_model.train_model()

app = Flask(__name__)

@app.route('/get_user_startup_info', methods=['GET', 'POST'])
def process_user_startup_info():
    if request.method == 'POST':
        clf_type = request.form(['clf_type'])
        category = request.form(['category'])
        total_funding = request.form(['total_funding'])
        city = request.form(['city'])
        funding_rounds = request.form(['funding_rounds'])
        founded_at = request.form(['founded_at'])
        first_funding_at = request.form(['first_funding_at'])
        last_funding_at = request.form(['last_funding_at'])

        accuracy_score, confusion_matrix, most_important_factors = startup_success_pred_model.run_diagnostics_model(clf_type)

        fate_of_startup = startup_success_pred_model.run_user_model(clf_type, category, total_funding, city, funding_rounds, founded_at, first_funding_at, last_funding_at)

        return render_template('model_gui_frontend.html', fate_of_startup=fate_of_startup, accuracy_score=accuracy_score, confusion_matrix=confusion_matrix, most_important_factors=most_important_factors)

    else:
        return render_template('model_gui_frontend.html')






