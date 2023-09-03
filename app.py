from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import pandas as pd

app = Flask(__name__)


def gen_sequence(id_df, seq_length, seq_cols):
    df_zeros = pd.DataFrame(np.zeros((seq_length - 1, id_df.shape[1])), columns=id_df.columns)
    # id_df=df_zeros.append(id_df,ignore_index=True)
    id_df = pd.concat([df_zeros, id_df], ignore_index=True)
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array = []
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        lstm_array.append(data_array[start:stop, :])
    return np.array(lstm_array)


# function to generate labels
def gen_label(id_df, seq_length, seq_cols, label):
    df_zeros = pd.DataFrame(np.zeros((seq_length - 1, id_df.shape[1])), columns=id_df.columns)
    # id_df=df_zeros.append(id_df,ignore_index=True)
    id_df = pd.concat([df_zeros, id_df], ignore_index=True)
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    y_label = []
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        y_label.append(id_df[label][stop])
    return np.array(y_label)


# Load your pre-trained LSTM model
setting_cols = ['setting_{}'.format(i + 1) for i in range(3)]
sensor_cols = ['sensor_{}'.format(i + 1) for i in range(21)]
feature_cols = setting_cols + sensor_cols
seq_length = 50
rul_seq_length = 30
seq_cols = feature_cols


# Define a function to predict failure probability
def predict_failure(machine_id, model, df_test):
    machine_df = df_test[df_test.engine_id == machine_id]
    machine_test = gen_sequence(machine_df, seq_length, seq_cols)
    m_pred = model.predict(machine_test, verbose=0)
    failure_prob = list(m_pred[-1] * 100)[0]
    return failure_prob


def predict_rul(machine_id, model_rul, df_test_rul):
    machine_df = df_test_rul[df_test_rul.engine_id == machine_id]
    rul_cols = ['setting_1', 'setting_2', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9',
                'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']
    machine_test = gen_sequence(machine_df, rul_seq_length, rul_cols)
    m_pred = model_rul.predict(machine_test, verbose=0)
    rul = list(m_pred[-1] * 1)[0]
    return int(rul)


# Define color-coding function
def get_color_code(probability):
    if probability < 20:
        return 'green'
    elif probability < 50:
        return 'yellow'
    else:
        return 'red'


# Define color-coding function
def get_rul_color(rul):
    if rul > 30:
        return 'green'
    elif 10 <= rul <= 30:
        return 'yellow'
    else:
        return 'red'


@app.route('/', methods=['GET', 'POST'])
def index():
    grid_data = []
    if request.method == 'POST':
        machine_id = int(request.form['machine_id'])
        model_path = 'models/model_pm_30'
        data_path = 'data/df_test.csv'
        model = tf.keras.models.load_model(model_path)
        df_test = pd.read_csv(data_path)
        failure_prob = predict_failure(machine_id, model, df_test)
        failure_probability = round(failure_prob, 2)
        color_code = get_color_code(failure_probability)
        if 'calculate_rul' in request.form:
            model_rul = tf.keras.models.load_model('models/model_calculate_rul')
            df_test_rul = pd.read_csv('data/df_rul_test.csv')
            # Call the predict_rul function to calculate RUL
            rul_result = predict_rul(machine_id, model_rul, df_test_rul)
            return render_template('index.html', machine_id=machine_id, probability=failure_probability,
                                   color_code=color_code, grid_data=grid_data, rul=rul_result)
        else:
            return render_template('index.html', machine_id=machine_id, probability=failure_probability,
                                   color_code=color_code, grid_data=grid_data)
    return render_template('index.html', grid_data=grid_data)


@app.route('/grid')
def grid():
    button_clicked = request.args.get('prediction_button')
    if button_clicked == '30':
        model_path = 'models/model_pm_30'
        data_path = 'data/df_test.csv'
    elif button_clicked == '20':
        model_path = 'models/model_pm_20'
        data_path = 'data/df_test_20.csv'
    elif button_clicked == '10':
        model_path = 'models/model_pm_10'
        data_path = 'data/df_test_10.csv'
    else:
        return "Invalid button clicked"
    grid_data = []
    model = tf.keras.models.load_model(model_path)
    df_test = pd.read_csv(data_path)
    for machine_id in range(1, 101):
        failure_prob = predict_failure(machine_id, model, df_test)
        failure_probability = round(failure_prob, 3)
        color_code = get_color_code(failure_probability)
        grid_data.append(
            {'machine_id': machine_id, 'color_code': color_code, 'failure_probability': failure_probability})
    green_count = sum(1 for item in grid_data if item['color_code'] == 'green')
    yellow_count = sum(1 for item in grid_data if item['color_code'] == 'yellow')
    red_count = sum(1 for item in grid_data if item['color_code'] == 'red')

    return render_template('grid.html', button_clicked=button_clicked, grid_data=grid_data, green_count=green_count,
                           yellow_count=yellow_count, red_count=red_count)


@app.route('/rul_grid')
def rul_grid():
    grid_data = []
    model_rul = tf.keras.models.load_model('models/model_calculate_rul')
    df_test_rul = pd.read_csv('data/df_rul_test.csv')

    for machine_id in range(1, 101):
        # Call the predict_rul function to calculate RUL
        rul_result = predict_rul(machine_id, model_rul, df_test_rul)
        rul_color_code = get_rul_color(rul_result)
        grid_data.append({'machine_id': machine_id, 'rul_color_code': rul_color_code, 'rul_result': rul_result})
    rul_green_count = sum(1 for item in grid_data if item['rul_color_code'] == 'green')
    rul_yellow_count = sum(1 for item in grid_data if item['rul_color_code'] == 'yellow')
    rul_red_count = sum(1 for item in grid_data if item['rul_color_code'] == 'red')

    return render_template('rul_grid.html', grid_data=grid_data, rul_green_count=rul_green_count,
                           rul_yellow_count=rul_yellow_count, rul_red_count=rul_red_count)


if __name__ == '__main__':
    app.run(debug=True)
