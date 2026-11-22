import streamlit as st
import pandas as pd
import time
import scikit_machine as sm


st.title(':blue[Watering] Actions Dataset ',icon=':material/water_ph:',text_alignment='center')
st.divider()

left,right = st.columns(2,vertical_alignment='center')

with left:
    st.subheader(':red[Five] :green[Layers] Formula:',text_alignment='left',icon=':material/subscript:',divider="rainbow")

with right:
    with st.popover(":red[First] :blue[Layer]"):
        st.markdown(':orange[First] :gray[Layer] :violet[Formula] :bell:')
        st.latex(r'''
                1. O^1_i(x₁) = \mu_{A_1}(x₁),O^1_i(x₂) = \mu_{B_1}(x₂),i = 1,2
                ''')
    with st.popover(":green[Second] :yellow[Layer]"):
        st.markdown(':gray[Second] :blue[Layer] :orange[Formula] :bell:')
        st.latex(r'''
                2.O^2_i = w_i = \mu_{A_1}(x_1)*\mu_{B_1}(x_2),i = 1,2
        ''')
    with st.popover(":gray[Third] :violet[Layer]"):
        st.markdown(':green[Third] :yellow[Layer] :red[Formula] :bell:')
        st.latex(r'''
                3.O^3_i = w_i   = w_i/(w_1 + w_2),i = 1,2
        ''')
    with st.popover(":yellow[Fourth] :blue[Layer]"):
        st.markdown(':gray[Fourth] :red[Layer] :blue[Formula] :bell:')
        st.latex(r'''
                4.O^4_i = w_if_i = w_i(p_1x_1 + q_1x_2 + r_i),i = 1,2
        ''')
    with st.popover(":violet[Fifth] :red[Layer]"):
        st.markdown(':blue[Fifth] :yellow[Layer] :gray[Formula] :bell:')
        st.latex(r'''
                5.O^5_i = w_if_i = w_if_i / w_i,1 = 1,2
        ''')
st.subheader(':red[Five] :green[Layers] :violet[Neural Image:]',text_alignment='left',icon=':material/layers:',divider="rainbow")
st.image('image/Neural.png', caption=':red[ANFIS] example with :green[two] :blue[inputs],:green[two] :violet[membership] functions in each :blue[input],and :green[two] :red[rules] :moyai:', alt='ANFIS Layer Image Example')
st.divider()
st.subheader(':green[Hybrid] :violet[Learning] :red[Process Image:]',text_alignment='left',icon=':material/published_with_changes:',divider="rainbow")

st.image('image/hybrid.png', caption=':blue[The] :green[Hybrid] :violet[learning] :red[process] :gray[implemented in ANFIS model] :moyai:', alt='Hybrid Learning Process Example')
st.divider()

st.subheader(':blue[Table] :green[Software] :gray[Of] :red[ANFIS]',text_alignment='left',icon=':material/code:',divider="rainbow")

@st.cache_data
def load_data():
    rdf = pd.DataFrame(
        {
            "Name":["ANFIS-C","ANFIS-Vignette","Matlab-ANFIS","ANFIS-PyTorch","ANFIS-Numpy","ANFIS-PSO","Scikit-ANFIS"],
            "Language":["C","R","Matlab","Python 3","Python 3","Python 3","Python 3"],
            "Library":[None,None,None,"PyTorch","Numpy","Numpy","PyTorch+Numpy"],
            "MFt":[4,4,12,5,3,3,12],
            "Learning strategy":["Hybrid/Online","Hybrid/Online","Hybrid/Online","Hybrid/Online","Hybrid","PSO","Hybrid/Online"],
            "Sklearn":["No","No","No","No","No","No","Yes"],
            "Release":["1993","2012","2023","2019","2020","2021","2023"]
        }
    )
    rdf = rdf.set_index("Name")
    return rdf

@st.cache_data
def load_more_data():
    rdf = pd.DataFrame(
        {
            "Name": ["Neuro-Fuzzy", "DNFC", "MBGD-RDA", "TSK-MBGD-UR-BN", "FCM-RDpA[", "HTSK-LN-ReLU",
                     "PyTSK","Scikit-ANFIS"],
            "Language": ["C", "Matlab", "Python 3", "Python 3", "Matlab", "Python 3", "Python 3","Python 3"],
            "MFt": [1, 1, 1, 1, 1, 1, 2,12],
            "Layers":[4,8,5,6,5,7,6,5],
            "Optimization method":["Gradient Descent","Gradient Descent","MBGD+AdaBound"," MBGD+AdaBound+BN+UR","MBGD+Powerball+AdaBelief","MBGD+Adam+LU+ReLU","MBGD+Adam","ANFIS"],
            "Tasks":["Classification","Classification","Regression","Classification","Regression","Regression","Classification","Regression+Classification"],
            "Sklearn":["No","No","No","No","No","Yes","Yes","Yes"],
            "Release":["1993","2020","2020","2020","2021","2022","2022","2023"]
        }
    )
    rdf = rdf.set_index("Name")
    return rdf

df = load_data()
dfm = load_more_data()
st.dataframe(df,placeholder="N/A",alt="software for ANFIS")
st.divider()

st.subheader(':green[The] :red[difference] :gray[between] :blue[Scikit-ANFIS] '
             'and other :red[deep neural Fuzzy] methods for two common tasks: :green[regression] and  :blue[classification]',text_alignment='left',icon=':material/difference:',divider="rainbow")
st.dataframe(dfm,lazy=True,alt="The different between other Scikit-ANFIS and other deep neural fuzzy methods")
st.divider()

st.subheader(':violet[Scikit]-:red[ANFIS] :green[architecture]',text_alignment='left',icon=':material/schema:',divider="rainbow")
st.image('scikit-anfis/skanfis_architecture.png', caption=':red[Overview] :gray[of] :violet[Scikit-ANFIS] :green[architecture] :moyai:', alt='Scikit-ANFIS architecture')
st.divider()

st.subheader(':green[Manual] :violet[Generation] :red[Method]',text_alignment='left',icon=':material/fiber_manual_record:',divider="rainbow")
st.image('image/manual.png', caption=':green[The] :gray[illustration of the] :red[manual generation method] :blue[for the general fuzzy system] :moyai:', alt='Scikit-ANFIS architecture')
st.divider()

st.subheader(':red[Automatic] :blue[method] :green[ANFIS] fuzzy system',text_alignment='left',icon=':material/auto_transmission:',divider="rainbow")
st.image('image/automatic.png', caption=':red[The illustration ] :gray[ of the] :blue[ automatic method for ] :violet[ANFIS fuzzysystem] :moyai:', alt='Automatic Method Scikit-ANFIS')
st.divider()

tab1, tab2, tab3 = st.tabs([":material/developer_guide: :rainbow[Pseudo Code]"," :material/code_blocks: :rainbow[Code Example]"," :moyai: :red[Result]"],
                           default=":material/developer_guide: :rainbow[Pseudo Code]",on_change="rerun")

_Pseudo_Explanation = r"""
The input, training, and test datasets are
denoted as :green-badge[X,Y], :blue-badge[ ðXtrain; YtrainÞ], and :red-badge[ðXtest; YtestÞ].The :red[main procedure] of :green[Scikit-ANFIS] 
:orange-badge[encapsulated] between :violet-badge[lines 1 and 15],comprise :red-badge[three key elements].:blue-badge[Firstly], in :violet-badge[line 1],
a general fuzzy system object **fs** is :red-badge[manually] created.Following that,:violet-badge[lines 2 and 3] specify :red[fuzzy sets],:red[input variables],
:red[fuzzy rules], and :red[output variables] for fs.In :violet-badge[line 4],we check whether **fs** is empty.If it is **not**,we use the :orange-badge[antecedent] and
:orange-badge[consequent] parameters along with :red[rules] from **fs** to represent a :blue-badge[5-layer] ANFIS model called **anfis_layers** in :violet-badge[line 5].
At the same time, we create a **scikit_anfis** object based on **anfis_layers** in :violet-badge[line 6],also specifying the **maximum number** of epochs for training
:orange-badge[(max_epoch)],the training strategy :orange-badge[(hybrid)],and the task type :orange-badge[(label)],and the **optimizer**.By default,:orange-badge[(max_epoch)]
is set to :red[10],:orange-badge[(hybrid)] is set to :green[True],indicating the use of the :orange-badge[hybrid] training strategy,**label** is set to **"r"**,indicating that the
:red[model] is intended for :blue-badge[regression] tasks,and :orange-badge[optimizer] is set to the **gradient descent** method namely :red["Adam"].:blue-badge[Secondly],if fs is **empty**, 
we automatically create  an ANFIS fuzzy system from :violet-badge[lines 7 to 10].:violet-badge[Line 8] is the :red-badge[5-layer] ANFIS model **anfis_layers derives from the 
input dataset :green-badge[(X,Y)],and :violet-badge[line 9] generates a **scikit_anfis** object and specifies its associated parameters,such as  :orange-badge[max_epoch],
:orange-badge[hybrid], :orange-badge[label],and :orange-badge[optimizer].:blue-badge[Thirdly],the most crucial step involves the implementation of the
ANFIS :orange-badge[optimizer] from  :violet-badge[lines 11 to 15]. At each epoch in the loop, we feed the training dataset  :blue-badge[ðXtrain; YtrainÞ] into the 
**scikit anfis** object to complete the forward pass from :violet-badge[layer 1 to layer 5] in its ANFIS model. During this process, we update the consequent parameters 
and obtain the final output **Y^**. We then compute the RMSE loss between the predicted **Y^** and the training target :blue-badge[Ytrain],
followed by updating the premise parameters of the ANFIS model in its **backward process**.
"""

_Second_Pseudo_Explanation = r"""
When the ANFIS  **optimizer** has completed training the :red[model], confidently use **fs** for fuzzy inference of test data
from :violet-badge[lines 16 to 20]. Return the premise and consequent
parameters to fs using the set :orange-badge[antecedentðÞ] and
set :orange-badge[consequentðÞ] interfaces. Then, directly call the :red[inference()] function of **fs** to complete the fuzzy system reasoning. 
Alternatively, we can opt for the more convenient **scikit anfis** object in :violet-badge[line 21] for fuzzy :red[inference]. This
method uses the test input data :red[Xtest] to generate the predicted output :red[Ypred] through fuzzy reasoning of the ANFIS
model. This option has been used in subsequent cases in this paper. It is important to :orange-badge[note] that during testing, there
is no need to provide the test target value :red-badge[Ytest] to the trained **scikit anfis** object since all of its parameters are already
fixed. Finally, in the last line, we compare and evaluate the fuzzy :red[inference] result :red[Ypred] with the test target value :red-badge[Ytest].
"""

@st.cache_data
def get_source_code():
    return  '''
from skanfis.fs import *
from skanfis import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import os

  
# !pip install kagglehub
   import kagglehub
# !pip install ipywidgets
path = kagglehub.dataset_download("ziya07/plant-health-data")
print("Path to dataset files:", path)
df = pd.read_csv(os.path.join(path, 'plant_health_data.csv'))
df = df.round(1)
print('Shape:', df.shape)
print('Columns:', list(df.columns))
print('First 5 rows:')
df.head()
print('Data types:')
df.dtypes
print('Basic statistics:')
df.describe()

df.head()

train_data, test_data = train_test_split(df, test_size=0.5, random_state=42)
y_test = test_data.pop('Plant_Health_Status')
print(y_test.head())
print(train_data.head())
train_data[["Soil_Moisture","Ambient_Temperature","Soil_Temperature","Humidity","Light_Intensity","Soil_pH","Nitrogen_Level","Phosphorus_Level","Potassium_Level","Chlorophyll_Content","Electrochemical_Signal"]].astype('float64')
test_data[["Soil_Moisture","Ambient_Temperature","Soil_Temperature","Humidity","Light_Intensity","Soil_pH","Nitrogen_Level","Phosphorus_Level","Potassium_Level","Chlorophyll_Content","Electrochemical_Signal"]].astype('float64')
df.dtypes

fs = FS()
# Encode target labels: map health status to watering actions
# High Stress -> Watering (2.0), Moderate Stress -> Small Watering (1.0), Healthy -> No Watering (0.0)
label_map = {'High Stress': 2.0, 'Moderate Stress': 1.0, 'Healthy': 0.0}
y_train = train_data['Plant_Health_Status'].map(label_map)
y_test_numeric = y_test.map(label_map)

# Map original labels to watering action names for display
action_map = {'High Stress': 'Watering', 'Moderate Stress': 'Small Watering', 'Healthy': 'No Watering'}
y_test_watering = y_test.map(action_map)
print('Training labels distribution:')
print(y_train.value_counts().rename(index={2.0: 'Watering', 1.0: 'Small Watering', 0.0: 'No Watering'}))
print('Target shape:', y_train.shape)

# Define input features and their ranges based on data statistics
input_features = [
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

def make_lv(name, low, mid, high):
    """Create a linguistic variable with low/medium/high triangular fuzzy sets."""
    low_fs = TriangleFuzzySet(a=low, b=low, c=mid, term='low')
    mid_fs = TriangleFuzzySet(a=low, b=mid, c=high, term='medium')
    high_fs = TriangleFuzzySet(a=mid, b=high, c=high, term='high')
    return LinguisticVariable([low_fs, mid_fs, high_fs])

# Add input linguistic variables
for fname, (low, mid, high) in input_features:
    fs.add_linguistic_variable(fname, make_lv(fname, low, mid, high))

print(f'Added {len(input_features)} input linguistic variables')

# For Sugeno ANFIS, the output is numeric - no need for output linguistic variables in the FS
# The output is handled by the scikit_anfis outvarnames parameter
print('No output LV needed for Sugeno ANFIS')

# Define simple rules: use a few representative rules covering key input patterns
# Rule 1: Low moisture + high temp -> high stress
R1 = 'IF (Soil_Moisture IS low) AND (Ambient_Temperature IS high) THEN (Plant_Health_Status IS high)'
# Rule 2: High moisture + medium temp -> healthy
R2 = 'IF (Soil_Moisture IS high) AND (Ambient_Temperature IS medium) THEN (Plant_Health_Status IS low)'
# Rule 3: Medium everything -> moderate stress
R3 = 'IF (Soil_Moisture IS medium) AND (Humidity IS medium) THEN (Plant_Health_Status IS medium)'
# Rule 4: Low nutrients -> high stress
R4 = 'IF (Nitrogen_Level IS low) AND (Chlorophyll_Content IS low) THEN (Plant_Health_Status IS high)'
# Rule 5: High nutrients + good light -> healthy
R5 = 'IF (Nitrogen_Level IS high) AND (Light_Intensity IS high) THEN (Plant_Health_Status IS low)'
print('Rules defined')

fs.add_rules([R1, R2, R3, R4, R5])
print('Rules added to fuzzy system')

#%%
# Prepare input data - select only the features defined in the fuzzy system
input_feature_names = [f[0] for f in input_features]
print('Input features:', input_feature_names)
print('Number of input features:', len(input_feature_names))

X_train = train_data[input_feature_names].astype('float64')
X_test = test_data[input_feature_names].astype('float64')
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)

model = scikit_anfis(fs, epoch=10, hybrid=True)
print('ANFIS model created')

# Train the model with both X and y
model.fit(X_train, y_train.values)
print('Model training complete!')

# Evaluate on test data
y_pred = model.predict(X_test)
if y_pred.ndim > 1:
    y_pred = y_pred.flatten()
mse = mean_squared_error(y_test_numeric.values, y_pred)
print(f'Test MSE: {mse:.4f}')
print(f'Test RMSE: {mse**0.5:.4f}')

# Show some predictions with watering action labels
import numpy as np
pred_labels = np.where(y_pred >= 1.5, 'Watering', np.where(y_pred >= 0.5, 'Small Watering', 'No Watering'))
results = pd.DataFrame({
    'Timestamp': pd.to_datetime(test_data['Timestamp'].values),
    'Actual_Action': y_test_watering.values,
    'Predicted_Action': pred_labels,
    'Predicted_Score': np.round(y_pred, 2)
})
results = results.sort_values('Timestamp').reset_index(drop=True)
print('Sample predictions (Watering Actions):')
print(results.head(10))
'''

@st.cache_data
def get_data():
    data = pd.read_csv('plant_health_data.csv')
    return data.to_csv(index=False).encode('utf-8')

@st.cache_data
def get_chart():
    anfis_data = sm.SCIKITMACHINES()
    return anfis_data.result

def stream_data():
    for word in _Pseudo_Explanation.split(" "):
        yield word + " "
        time.sleep(0.1)

    for word in _Second_Pseudo_Explanation.split(" "):
        yield word + " "
        time.sleep(0.1)

if tab1.open:
    with tab1:
        st.image('image/pseudo.png',
                 caption=':gray[Pseudo Code for Scikit-ANFIS] Implementation :moyai:',
                 alt='Pseudo Code Image ')
        st.write_stream(stream_data)

if tab2.open:
    with tab2:
        st.code(get_source_code(),line_numbers=True)
        st.sidebar.download_button(
            label='Download Dataset',
            data=get_data(),
            file_name='plant_health_data.csv',
            mime='text/csv',
            icon=":material/download:",
            shortcut="Enter"
        )

if tab3.open:
    with tab3:
        st.line_chart(data=get_chart())
        with open("scikit_anfis.pdf",'rb') as f:
            st.sidebar.download_button(
                label='Download Journal',
                data=f,
                file_name='scikit_anfis.pdf',
                mime='application/pdf',
                icon=":material/download:",
                shortcut="Enter"
            )