from skanfis.fs import *
from skanfis import scikit_anfis
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np


class SCIKITMACHINES:

    def __init__(self):
        self.df, self.data = self.get_data
        self.label_map = {'High Stress': 2.0, 'Moderate Stress': 1.0, 'Healthy': 0.0}
        self.action_map = {'High Stress': 'Watering', 'Moderate Stress': 'Small Watering', 'Healthy': 'No Watering'}
        self.train_data, self.test_data = self.make_train_data()
        self.y_train, self.y_test_numeric, self.y_test_watering, self.y_test = self.make_y_variable()
        self.x_train, self.x_test = self.make_x_variable()
        self.fizz = FS()
        self.features = self.input_features
        self.rule = self.fuzzy_rules
        self.set_linguistic_variable()
        self.add_rules()
        self.model = self.make_model()
        self.mse, self.y_pred = self.set_mse_pred()
        self.result = self.set_result()


    @staticmethod
    def make_lv(**kwargs):
        low_fs = TriangleFuzzySet(a=kwargs['low'], b=kwargs['low'], c=kwargs['mid'], term='low')
        mid_fs = TriangleFuzzySet(a=kwargs['low'], b=kwargs['mid'], c=kwargs['high'], term='medium')
        high_fs = TriangleFuzzySet(a=kwargs['mid'], b=kwargs['high'], c=kwargs['high'], term='high')
        return LinguisticVariable([low_fs, mid_fs, high_fs])

    def set_linguistic_variable(self):
        for fname, (low, mid, high) in self.features:
            self.fizz.add_linguistic_variable(fname, self.make_lv(low=low, mid=mid, high=high))
        print(f'Added {len(self.features)} input linguistic variables')

    def add_rules(self):
        self.fizz.add_rules([values for key,values in self.rule.items()])
        print('Rules added to fuzzy system')

    @staticmethod
    def check_data(data: pd.DataFrame):
        print('Shape:',data.shape)
        print('Columns:', list(data.columns))
        print('First 5 rows:')
        print(data.head())
        print('Data types:')
        print(data.dtypes)

    @staticmethod
    def check_fs(**kwargs):
        print(kwargs['y_test'].head())
        print(kwargs['train_data'].head())
        print(kwargs['data'].dtypes)
        print('Training labels distribution:')
        print(kwargs['y_train'].value_counts().rename(index={2.0: 'Watering', 1.0: 'Small Watering', 0.0: 'No Watering'}))
        print('Target shape:', kwargs['y_train'].shape)
        print('X_train shape:', kwargs['x_train'].shape)
        print('X_test shape:', kwargs['x_test'].shape)

    @property
    def get_data(self):
        data = pd.read_csv('plant_health_data.csv')
        df = data.round(1)
        return df ,data

    def make_model(self):
        model = scikit_anfis(self.fizz, epoch=10, hybrid=True)
        print('ANFIS model created')
        model.fit(self.x_train, self.y_train.values)
        print('Model training complete!')
        return model

    def set_mse_pred(self):
        y_pred = self.model.predict(self.x_test)
        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()
        mse = mean_squared_error(self.y_test_numeric.values, y_pred)
        print(f'Test MSE: {mse:.4f}')
        print(f'Test RMSE: {mse ** 0.5:.4f}')
        return mse, y_pred

    def set_result(self):
        pred_labels = np.where(self.y_pred >= 1.5, 'Watering', np.where(self.y_pred >= 0.5, 'Small Watering', 'No Watering'))
        results = pd.DataFrame({
            'Timestamp': pd.to_datetime(self.test_data['Timestamp'].values),
            'Actual_Action': self.y_test_watering.values,
            'Predicted_Action': pred_labels,
            'Predicted_Score': np.round(self.y_pred, 2)
        })
        results = results.sort_values('Timestamp').reset_index(drop=True)
        print('Sample predictions (Watering Actions):')
        print(results.head(10))
        return results

    def make_train_data(self):
        train_data, test_data = train_test_split(self.df, test_size=0.5, random_state=42)
        cols = ["Soil_Moisture", "Ambient_Temperature", "Soil_Temperature", "Humidity", "Light_Intensity", "Soil_pH",
                "Nitrogen_Level", "Phosphorus_Level", "Potassium_Level", "Chlorophyll_Content",
                "Electrochemical_Signal"]
        train_data[cols] = train_data[cols].astype('float64')
        test_data[cols] = test_data[cols].astype('float64')
        return train_data, test_data

    def make_y_variable(self):
        y_test = self.test_data.pop('Plant_Health_Status')
        y_train = self.train_data['Plant_Health_Status'].map(self.label_map)
        y_test_numeric = y_test.map(self.label_map)
        y_test_watering = y_test.map(self.action_map)
        return y_train,y_test_numeric, y_test_watering,y_test

    def make_x_variable(self):
        x_train = self.train_data[[f[0] for f in self.input_features]].astype('float64')
        x_test = self.test_data[[f[0] for f in self.input_features]].astype('float64')
        return x_train,x_test

    @property
    def input_features(self):
        return  [
            ('Soil_Moisture', (10, 25, 40)),
            ('Ambient_Temperature', (18, 24, 30)),
            ('Soil_Temperature', (15, 20, 25)),
            ('Humidity', (40, 55, 70)),
            ('Light_Intensity', (200, 600, 1000)),
            ('Soil_pH', (5.5, 6.5, 7.5)),
            ('Nitrogen_Level', (10, 30, 50)),
            ('Phosphorus_Level', (10, 30, 50)),
            ('Potassium_Level', (10, 30, 50)),
            ('Chlorophyll_Content', (20, 35, 50)),
            ('Electrochemical_Signal', (0, 1, 2))
        ]

    @property
    def fuzzy_rules(self):
        print('Rules defined')
        return {
            'R1':'IF (Soil_Moisture IS low) AND (Ambient_Temperature IS high) THEN (Plant_Health_Status IS high)',
            'R2':'IF (Soil_Moisture IS high) AND (Ambient_Temperature IS medium) THEN (Plant_Health_Status IS low)',
            'R3':'IF (Soil_Moisture IS medium) AND (Humidity IS medium) THEN (Plant_Health_Status IS medium)',
            'R4':'IF (Nitrogen_Level IS low) AND (Chlorophyll_Content IS low) THEN (Plant_Health_Status IS high)',
            'R5':'IF (Nitrogen_Level IS high) AND (Light_Intensity IS high) THEN (Plant_Health_Status IS low)'
        }

if __name__ == '__main__':
    anfis_test: SCIKITMACHINES = SCIKITMACHINES()
    anfis_test.check_data(anfis_test.data)
    anfis_test.check_fs(y_test=anfis_test.y_test,train_data=anfis_test.train_data,
                        data=anfis_test.data,y_train=anfis_test.y_train,x_train=anfis_test.x_train,x_test=anfis_test.x_test)
