from model_gui_backend import StartupSuccessPredModel
from dataset import load_dataset
from flask import Flask, request, render_template

startup_success_pred_model = StartupSuccessPredModel(load_dataset())
startup_success_pred_model.run_train_model()

app = Flask(__name__)

@app.route('/')
def index():
    industries_list, funding_rounds_options, years = startup_success_pred_model.get_necessary_choices()
    return render_template('model_gui_frontend.html', industries_list=industries_list, funding_rounds_options=funding_rounds_options, years=years)


@app.route('/get_user_startup_info', methods=['GET', 'POST'])
def process_user_startup_info():
    if request.method == 'POST':
        clf_type = request.form.getlist('clf_type')[0] # Select first and only element to 'delistify' the list into a regular variable
        category = request.form.getlist('category')[0]
        total_funding = request.form.getlist('total_funding')[0]
        city = request.form.getlist('city')[0]
        funding_rounds = request.form.getlist('funding_rounds')[0]
        founded_at = request.form.getlist('founded_at')[0]
        first_funding_at = request.form.getlist('first_funding_at')[0]
        last_funding_at = request.form.getlist('last_funding_at')[0]

        accuracy_score, confusion_matrix, most_important_factors = startup_success_pred_model.run_diagnostics_model(clf_type)

        accuracy_score = f'{round(accuracy_score, 4) * 100}%'
        most_important_factors = [[most_important_factors[i][j] for j in range(2)] for i in range(len(most_important_factors))]
        
        for factor in most_important_factors:
            factor[0] = ' '.join(factor[0].split('_')).title()
            factor[1] = f'{abs(round(factor[1], 4) * 100)}%'

        fate_of_startup = startup_success_pred_model.run_user_model(clf_type, category, total_funding, city, funding_rounds, founded_at, first_funding_at, last_funding_at)

        startup_success_pred_model.clear_log_file('startup_model_feedback.log')

        industries_list, funding_rounds_options, years = startup_success_pred_model.get_necessary_choices()

        return render_template('model_gui_frontend.html', fate_of_startup=fate_of_startup, accuracy_score=accuracy_score, confusion_matrix=confusion_matrix, most_important_factors=most_important_factors, industries_list=industries_list, funding_rounds_options=funding_rounds_options, years=years, category=category, total_funding=total_funding, city=city, funding_rounds=funding_rounds, founded_at=founded_at, first_funding_at=first_funding_at, last_funding_at=last_funding_at)

    else:
        industries_list, funding_rounds_options, years = startup_success_pred_model.get_necessary_choices()
        return render_template('model_gui_frontend.html', industries_list=industries_list, funding_rounds_options=funding_rounds_options, years=years)


if __name__ == '__main__':
    app.run(debug=True)
