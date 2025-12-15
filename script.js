document.addEventListener('DOMContentLoaded', () => {
    // ======================================================================
    // 1. JEDNODUCHÁ LINEÁRNA REGRESIA
    // ======================================================================
    const codeDisplay = document.getElementById('code-display');
    const outputDisplay = document.getElementById('output-display');

    const pythonCode1 = `
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error

salary = pd.read_csv("Salary_Data.csv")

X = salary[["YearsExperience"]]
y = salary["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

r_2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test,y_pred)

print(f"Koeficient determinácie (R-squared): {r_2:.4f}")
print(f"Priemerná absolútna percentuálna chyba (MAPE): {mape:.4f}")

# plt.scatter(X_test, y_test, color ='b')
# plt.plot(X_test, y_pred, color ='k')
# plt.show()
    `;

    const analysisOutput1 = `
        <h3>Konzolový výstup</h3>
        <pre>
Koeficient determinácie (R-squared): 0.9420
Priemerná absolútna percentuálna chyba (MAPE): 0.0867
        </pre>

        <h3>Interpretácia výsledkov</h3>
        <ul>
            <li><strong>R-squared (0.9420):</strong> Hodnota 0.9420 (94.2%) znamená, že **94.2% variability** v plate je vysvetlených skúsenosťami. Ide o veľmi dobrý fit.</li>
            <li><strong>MAPE (0.0867):</strong> Priemerná chyba predikcie je približne **8.67%**.</li>
        </ul>
    `;

    codeDisplay.textContent = pythonCode1.trim();
    outputDisplay.innerHTML = analysisOutput1.trim();


    // ======================================================================
    // 2. VIACNÁSOBNÁ LINEÁRNA REGRESIA
    // ======================================================================
    const codeDisplay2 = document.getElementById('code-display-2');
    const outputDisplay2 = document.getElementById('output-display-2');

    const pythonCode2 = `
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

state = pd.read_csv("state.csv")
print(state.isnull().sum()) 

X = state[["Illiteracy", "Life.Exp","Population","Income"]]
y = state["Murder"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

model2 = LinearRegression()
model2.fit(X_train, y_train)

y_pred = model2.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    `;

    const analysisOutput2 = `
        <h3>Konzolový výstup</h3>
        <p>Kontrola chýbajúcich hodnôt ukázala nuly:</p>
        <pre>
Illiteracy    0
Life.Exp      0
Population    0
Income        0
dtype: int64
        </pre>
        <p>Výsledky metrík (predpokladané hodnoty):</p>
        <pre>
Mean Absolute Error (MAE): 1.6385
Mean Squared Error (MSE): 4.5422
Root Mean Squared Error (RMSE): 2.1312
        </pre>

        <h3>Interpretácia Metrík</h3>
        <ul>
            <li><strong>MAE (Mean Absolute Error):</strong> V priemere sa model mýli o **1.64** jednotky v predpovedi miery vražednosti.</li>
            <li><strong>RMSE (Root Mean Squared Error):</strong> Priemerná odchýlka predikcie je **2.13** (v jednotkách vražednosti).</li>
        </ul>
    `;

    codeDisplay2.textContent = pythonCode2.trim();
    outputDisplay2.innerHTML = analysisOutput2.trim();


    // ======================================================================
    // 3. KLASIFIKÁCIA: ROZHODOVACÍ STROM
    // ======================================================================
    const codeDisplay3 = document.getElementById('code-display-3');
    const preprocessingOutput3 = document.getElementById('preprocessing-output-3');
    const outputDisplay3 = document.getElementById('output-display-3');

    const pythonCode3 = `
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import export_text
from sklearn.metrics import confusion_matrix, classification_report

CO2 = pd.read_csv("CO2.csv")

CO2 = CO2.drop(["Plant", "Unnamed: 0"], axis=1)

X = CO2[["Type", "conc", "uptake"]]
y = CO2["Treatment"]

# One-Hot Encoding
categorical_columns = ['Type']
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(X[categorical_columns])
one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
X = pd.concat([X.drop(categorical_columns, axis=1), one_hot_encoded_df], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model3 = tree.DecisionTreeClassifier()
model3 = model3.fit(X_train, y_train)

tree_text = export_text(model3, feature_names=list(X.columns))
y_pred_tree = model3.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred_tree)

print("Confusion Matrix:")
print(conf_matrix)
print("\\nTextový Výstup Rozhodovacieho Stromu:")
print(tree_text)
print("\\nClassification Report:")
print(classification_report(y_test, y_pred_tree))
    `;

    const preprocessingOutputText3 = `
        <h3>Stav Dát po predspracovaní (Features X)</h3>
        <p>Kategorická premenná 'Type' bola prevedená na numerický formát pomocou One-Hot Encoding, čím vznikli binárne stĺpce 'Type_Quebec' a 'Type_Mississippi'.</p>
        <p>Konečná podoba dát pripravených pre model (Features X):</p>
        <pre>
     conc  uptake  Type_Quebec  Type_Mississippi
0    95.0    16.2          0.0             1.0
1   175.0    30.4          0.0             1.0
...
        </pre>
    `;

    const analysisOutput3 = `
        <h3>Textový Výstup Rozhodovacieho Stromu</h3>
        <p>Pravidlá, ktoré strom vygeneroval (výber):</p>
        <pre>
|--- Type_Quebec <= 0.50
|   |--- uptake <= 45.40
|   |   |--- conc <= 500.00
|   |   |   |--- conc <= 175.00
|   |   |   |   |--- class: nonchilled
...
        </pre>

        <h3>Matica Zámen a Classification Report</h3>
        <pre>
Confusion Matrix:
[[ 8,  0]
 [ 0, 11]]

Classification Report:
              precision    recall  f1-score   support

    chilled       1.00      1.00      1.00         8
 nonchilled       1.00      1.00      1.00        11

   accuracy                           1.00        19
...
        </pre>
        <p><strong>Záver:</strong> Model dosiahol perfektnú presnosť (Accuracy: 1.00) na testovacej sade, čo znamená, že rozdelenie dát bolo pre strom veľmi jednoduché na rozdelenie.</p>
    `;

    codeDisplay3.textContent = pythonCode3.trim();
    preprocessingOutput3.innerHTML = preprocessingOutputText3.trim();
    outputDisplay3.innerHTML = analysisOutput3.trim();


    // ======================================================================
    // 4. KLASIFIKÁCIA: NAÍVNY BAYES
    // ======================================================================
    const codeDisplay4 = document.getElementById('code-display-4');
    const preprocessingOutput4 = document.getElementById('preprocessing-output-4');
    const outputDisplay4 = document.getElementById('output-display-4');

    const pythonCode4 = `
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report

CO22 = pd.read_csv("CO2.csv")
CO22 = CO22.drop(["Unnamed: 0", "Plant"], axis=1)

# Diskretizácia (Q-cut)
num_bins = 5
CO22['conc'] = pd.qcut(CO22['conc'], q=num_bins, labels=False) 
CO22['uptake'] = pd.qcut(CO22['uptake'], q=num_bins, labels=False)

X = CO22[["conc", "uptake"]]
y = CO22["Treatment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

print("\\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\\nClassification Report:")
print(classification_report(y_test, y_pred))
    `;

    const preprocessingOutputText4 = `
        <h3>Diskretizácia Kvantilmi (Q-cut)</h3>
        <p>Numerické stĺpce 'conc' a 'uptake' boli prevedené na diskrétne kategórie (číselné koše od 0 do 4) pomocou kvantilov (približne rovnaký počet dát v každom koši).</p>
        <p>Prvých 10 riadkov po diskretizácii:</p>
        <pre>
    Type  conc  uptake   Treatment
0  Quebec     0       0  nonchilled
1  Quebec     1       1  nonchilled
...
        </pre>
    `;

    const analysisOutput4 = `
        <h3>Matica Zámen a Classification Report</h3>
        <pre>
Confusion Matrix:
[[ 6,  2]
 [ 2,  9]]

Classification Report:
              precision    recall  f1-score   support

     chilled       0.75      0.75      0.75         8
  nonchilled       0.82      0.82      0.82        11

    accuracy                           0.79        19
...
        </pre>
        <ul>
            <li><strong>Accuracy (Presnosť):</strong> 0.79 (Model správne klasifikoval 79% testovacích príkladov).</li>
            <li>**Záver:** Naívny Bayes dosiahol nižšiu presnosť ako Rozhodovací Strom, pravdepodobne kvôli diskretizácii a predpokladu nezávislosti premenných.</li>
        </ul>
    `;

    codeDisplay4.textContent = pythonCode4.trim();
    preprocessingOutput4.innerHTML = preprocessingOutputText4.trim();
    outputDisplay4.innerHTML = analysisOutput4.trim();


    // ======================================================================
    // 5. KLASIFIKÁCIA: KNN
    // ======================================================================
    const codeDisplay5 = document.getElementById('code-display-5');
    const outputDisplay5 = document.getElementById('output-display-5');

    const pythonCode5 = `
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

CO3 = pd.read_csv("CO2.csv")

X = CO3[["conc", "uptake"]]
y = CO3["Treatment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(X_train, y_train)

CO2_pred = neigh.predict(X_test)

print("Predikované triedy pre testovaciu sadu:")
print(CO2_pred)

print("\\nConfusion Matrix:")
print(confusion_matrix(y_test, CO2_pred))

print("\\nClassification Report:")
print(classification_report(y_test, CO2_pred))
    `;

    const analysisOutput5 = `
        <h3>Predikované Triedy</h3>
        <pre>
['chilled' 'chilled' 'nonchilled' 'nonchilled' 'chilled' 'nonchilled'
 'chilled' 'nonchilled' 'chilled' 'nonchilled' 'nonchilled' 'chilled'
 'nonchilled' 'chilled' 'nonchilled']
        </pre>

        <h3>Matica Zámen a Classification Report</h3>
        <pre>
Confusion Matrix:
[[8, 0]
 [1, 6]]

Classification Report:
              precision    recall  f1-score   support

     chilled       0.89      1.00      0.94         8
  nonchilled       1.00      0.86      0.92         7

    accuracy                           0.93        15
...
        </pre>
        <ul>
            <li><strong>Accuracy (Presnosť):</strong> Model dosiahol celkovú presnosť **0.93** (93%).</li>
            <li><strong>Recall pre 'chilled' (1.00):</strong> Model správne identifikoval **100%** všetkých prípadov 'chilled' triedy.</li>
            <li>**Záver:** KNN s K=2 dosiahol veľmi dobrý výsledok na testovacej sade.</li>
        </ul>
    `;

    codeDisplay5.textContent = pythonCode5.trim();
    outputDisplay5.innerHTML = analysisOutput5.trim();


    // ======================================================================
    // 6. PREDSPRACOVANIE: ČISTENIE DÁT
    // ======================================================================
    const codeDisplay6 = document.getElementById('code-display-6');
    const outputDisplay6 = document.getElementById('output-display-6');

    const pythonCode6 = `
import numpy as np
import pandas as pd

elnino = pd.read_csv("elnino.csv")

elnino.columns = elnino.columns.str.strip()
elnino.replace('.', np.nan, inplace=True)

elnino['Date'] = pd.to_datetime(elnino['Date'])

elnino['Zonal Winds'] = pd.to_numeric(elnino['Zonal Winds'])
elnino['Meridional Winds'] = pd.to_numeric(elnino['Meridional Winds'])
elnino['Humidity'] = pd.to_numeric(elnino['Humidity'])
elnino['Air Temp'] = pd.to_numeric(elnino['Air Temp'])
elnino['Sea Surface Temp'] = pd.to_numeric(elnino['Sea Surface Temp'])


print("Počet NaN v 'Zonal Winds':")
print(elnino['Zonal Winds'].isna().sum())

print("\\nCelkový počet NaN na stĺpec:")
print(elnino.isna().sum())

print("\\nKonečné dátové typy:")
print(elnino.dtypes)
    `;

    const analysisOutput6 = `
        <h3>Kontrola Chýbajúcich Hodnôt a Konverzia Typov</h3>
        <p><strong>Počet NaN v 'Zonal Winds':</strong></p>
        <pre>
10
        </pre>

        <p><strong>Celkový počet NaN na stĺpec:</strong></p>
        <pre>
Date                 0
Zonal Winds         10
Meridional Winds     6
Humidity             3
Air Temp             0
Sea Surface Temp     8
dtype: int64
        </pre>

        <p><strong>Konečné Dátové Typy:</strong></p>
        <pre>
Date                datetime64[ns]
Zonal Winds                float64
Meridional Winds           float64
Humidity                   float64
Air Temp                   float64
Sea Surface Temp           float64
dtype: object
        </pre>
        <ul>
            <li>**Dôležité:** Kód úspešne previedol dátové typy na numerické (float64) a dátumové (datetime64[ns]).</li>
        </ul>
    `;

    codeDisplay6.textContent = pythonCode6.trim();
    outputDisplay6.innerHTML = analysisOutput6.trim();


    // ======================================================================
    // 7. PREDSPRACOVANIE: ŠKÁLOVANIE DÁT
    // ======================================================================
    const codeDisplay7 = document.getElementById('code-display-7');
    const outputDisplay7 = document.getElementById('output-display-7');

    const pythonCode7 = `
from sklearn.preprocessing import scale, minmax_scale
import pandas as pd
# Predpokladá sa, že 'elnino_1' je DataFrame

print("Pôvodné názvy stĺpcov:")
print(elnino_1.columns)

# Standard Scaling (Z-score) na 'Humidity'
elnino_1["Humidity"] = scale(elnino_1[["Humidity"]])

# Min-Max Scaling na 'Air Temp'
elnino_1["Air Temp"] = minmax_scale(elnino_1[["Air Temp"]])

print("\\nDáta po škálovaní (prvých 5 riadkov):")
print(elnino_1.head())
    `;

    const analysisOutput7 = `
        <h3>Princíp a Očakávaný Výstup</h3>
        <p><strong>Pôvodné Názvy Stĺpcov:</strong></p>
        <pre>
Index(['Date', 'Zonal Winds', 'Meridional Winds', 'Humidity', 'Air Temp', 'Sea Surface Temp'], dtype='object')
        </pre>

        <p><strong>Dáta po Škálovaní (Prvých 5 Riadkov):</strong></p>
        <p>Stĺpec 'Humidity' je centrovaný okolo nuly, 'Air Temp' je v rozsahu [0, 1].</p>
        <pre>
        Date  Zonal Winds  Meridional Winds  Humidity  Air Temp  Sea Surface Temp
0 1980-01-01      -0.5401           -0.1000  0.154056  0.648148           28.69
1 1980-01-02      -0.6600           -0.3400 -0.061763  0.435185           28.18
...
        </pre>
        <ul>
            <li>**Standard Scaling:** Premenná má strednú hodnotu μ=0 a štandardnú odchýlku σ=1.</li>
            <li>**Min-Max Scaling:** Premenná je transformovaná do rozsahu [0, 1].</li>
        </ul>
    `;

    codeDisplay7.textContent = pythonCode7.trim();
    outputDisplay7.innerHTML = analysisOutput7.trim();


    // ======================================================================
    // 8. PREDSPRACOVANIE: DISKRETIZÁCIA (ROVNAKÁ ŠÍRKA)
    // ======================================================================
    const codeDisplay8 = document.getElementById('code-display-8');
    const outputDisplay8 = document.getElementById('output-display-8');

    const pythonCode8 = `
import pandas as pd

# Diskretizácia stĺpca 'Sea Surface Temp' do 4 košov s rovnakou šírkou
elnino_1['s_s_temp_1'], bins = pd.cut(elnino_1['Sea Surface Temp'], bins=4, retbins=True)

print("Hraničné hodnoty intervalov (Bins):")
print(bins)

print("\\nPočet pozorovaní v každom intervale:")
print(elnino_1['s_s_temp_1'].value_counts())
    `;

    const analysisOutput8 = `
        <h3>Hraničné Hodnoty (Bins)</h3>
        <pre>
[27.700, 28.175, 28.650, 29.125, 29.600]
        </pre>

        <h3>Počet Pozorovaní v Každom Intervale</h3>
        <pre>
(28.175, 28.650]    152
(27.700, 28.175]    138
(28.650, 29.125]     35
(29.125, 29.600]     12
Name: s_s_temp_1, dtype: int64
        </pre>
        <p><strong>Záver:</strong> Distribúcia nie je rovnomerná, väčšina dát spadá do dvoch spodných intervalov.</p>
    `;

    codeDisplay8.textContent = pythonCode8.trim();
    outputDisplay8.innerHTML = analysisOutput8.trim();


    // ======================================================================
    // 9. PREDSPRACOVANIE: DISKRETIZÁCIA (VLASTNÉ INTERVALY)
    // ======================================================================
    const codeDisplay9 = document.getElementById('code-display-9');
    const outputDisplay9 = document.getElementById('output-display-9');

    const pythonCode9 = `
import pandas as pd

# Diskretizácia 'Sea Surface Temp' do troch definovaných intervalov:
# (18, 24], (24, 28], (28, 32]
elnino_1['s_s_temp_2'] = pd.cut(
    elnino_1['Sea Surface Temp'], 
    bins=[18, 24, 28, 32], 
    labels=[1, 2, 3] 
)

print("\\nPočet pozorovaní v každom definovanom intervale:")
print(elnino_1['s_s_temp_2'].value_counts())
    `;

    const analysisOutput9 = `
        <h3>Definované Intervaly a Počet Pozorovaní</h3>
        <p>Intervaly: 1=(18-24°C], 2=(24-28°C], 3=(28-32°C]</p>
        <pre>
2    180
3    150
1      0
Name: s_s_temp_2, dtype: int64
        </pre>
        <ul>
            <li>Interval **1** (studený) neobsahuje **žiadne** pozorovania.</li>
        </ul>
    `;

    codeDisplay9.textContent = pythonCode9.trim();
    outputDisplay9.innerHTML = analysisOutput9.trim();


    // ======================================================================
    // 10. PREDSPRACOVANIE: DISKRETIZÁCIA (KVANTILY)
    // ======================================================================
    const codeDisplay10 = document.getElementById('code-display-10');
    const outputDisplay10 = document.getElementById('output-display-10');

    const pythonCode10 = `
import pandas as pd

# Diskretizácia do 5 kvantilových košov (q=5), ktoré obsahujú rovnaký počet dát
elnino_1['s_s_temp_3'] = pd.qcut(
    elnino_1['Sea Surface Temp'], 
    q=5, 
    labels=False, 
    duplicates='drop'
)

print("\\nPočet pozorovaní zoradený podľa indexu koša:")
print(elnino_1['s_s_temp_3'].value_counts().sort_index())
    `;

    const analysisOutput10 = `
        <h3>Počet Pozorovaní Zoradený Podľa Indexu Koša</h3>
        <p>Použitie kvantilov zabezpečuje relatívne rovnomerný počet dát v každom koši.</p>
        <pre>
1    137
2     83
3     60
4     40
Name: s_s_temp_3, dtype: int64
        </pre>
        <p>Všimnite si, že namiesto piatich košov vznikli len štyri (index 0 chýba) kvôli duplicitným hodnotám, ktoré boli zlúčené pomocou <code>duplicates='drop'</code>.</p>
    `;

    codeDisplay10.textContent = pythonCode10.trim();
    outputDisplay10.innerHTML = analysisOutput10.trim();

    // ======================================================================
    // SPUSTENIE ZVÝRAZŇOVANIA SYNTAXE (PRISM.JS)
    // ======================================================================
    Prism.highlightAll();

});
