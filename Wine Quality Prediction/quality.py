import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        wine_dataset = pd.read_csv(file_path)
        return wine_dataset
    else:
        return None

def preprocess_data(wine_dataset):
    
    wine_dataset['GoodQuality'] = np.where(wine_dataset['quality'] >= 6, 1, 0)
    X = wine_dataset.drop(['quality', 'GoodQuality'], axis=1)
    y = wine_dataset['GoodQuality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train_scaled, y_train):
    model = RandomForestClassifier()
    model.fit(X_train_scaled, y_train)
    return model

def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def on_predict_click(X_test_scaled, scaler):
    input_data = [float(entry.get()) for entry in entry_fields]
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)
    result_label.config(text=f"Prediction: {'Good Quality Wine' if prediction[0] == 1 else 'Bad Quality Wine'}")

def on_load_data_click():
    global model, scaler, X_test_scaled
    wine_dataset = load_data()
    if wine_dataset is not None:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(wine_dataset)
        model = train_model(X_train_scaled, y_train)
        accuracy = evaluate_model(model, X_test_scaled, y_test)
        messagebox.showinfo("Info", f"Data loaded and model trained successfully.\nAccuracy: {accuracy:.2f}")

def main():
    global entry_fields, result_label, model, scaler, X_test_scaled
    root = tk.Tk()
    root.title("Wine Quality Prediction")


    root.configure(bg='#f0f0f0')

    entry_fields = []
    attributes = ['Fixed Acidity', 'Volatile Acidity', 'Citric Acid', 'Residual Sugar', 'Chlorides',
                  'Free Sulfur Dioxide', 'Total Sulfur Dioxide', 'Density', 'pH', 'Sulphates', 'Alcohol']
    for i, attribute in enumerate(attributes):
        label = tk.Label(root, text=attribute, bg='#f0f0f0', fg='#333', font=('Arial', 12, 'bold'))
        label.grid(row=i, column=0, padx=10, pady=5, sticky='e')
        entry = tk.Entry(root, width=10, font=('Arial', 12))
        entry.grid(row=i, column=1, padx=10, pady=5, sticky='w')
        entry_fields.append(entry)

    predict_button = tk.Button(root, text="Predict", command=lambda: on_predict_click(X_test_scaled, scaler),
                                font=('Arial', 12), bg='#4CAF50', fg='#fff', padx=10)
    predict_button.grid(row=len(entry_fields), columnspan=2, pady=10)

    load_button = tk.Button(root, text="Load Data", command=on_load_data_click,
                             font=('Arial', 12), bg='#007BFF', fg='#fff', padx=10)
    load_button.grid(row=len(entry_fields)+1, columnspan=2, pady=10)

    result_label = tk.Label(root, text="", bg='#f0f0f0', font=('Arial', 14, 'bold'))
    result_label.grid(row=len(entry_fields)+2, columnspan=2, pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()


#GOOD
# 7.8,0.58,0.02,2.0,0.073,9.0,18.0,0.9968,3.36,0.57,9.5,7

#BAD
# 7.4,0.59,0.08,4.4,0.086,6.0,29.0,0.9974,3.38,0.5,9.0,4



# Using SVM


# import tkinter as tk
# from tkinter import filedialog, messagebox
# import pandas as pd
# import numpy as np
# from sklearn.svm import SVC  # Importing Support Vector Classifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# def load_data():
#     file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
#     if file_path:
#         wine_dataset = pd.read_csv(file_path)
#         return wine_dataset
#     else:
#         return None

# def preprocess_data(wine_dataset):
#     # Binary classification for wine quality
#     wine_dataset['GoodQuality'] = np.where(wine_dataset['quality'] >= 6, 1, 0)
#     X = wine_dataset.drop(['quality', 'GoodQuality'], axis=1)
#     y = wine_dataset['GoodQuality']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# def train_model(X_train_scaled, y_train):
#     model = SVC()  # Creating Support Vector Classifier
#     model.fit(X_train_scaled, y_train)
#     return model

# def evaluate_model(model, X_test_scaled, y_test):
#     y_pred = model.predict(X_test_scaled)
#     accuracy = accuracy_score(y_test, y_pred)
#     return accuracy

# def on_predict_click(X_test_scaled, scaler):
#     input_data = [float(entry.get()) for entry in entry_fields]
#     input_scaled = scaler.transform([input_data])
#     prediction = model.predict(input_scaled)
#     result_label.config(text=f"Prediction: {'Good Quality Wine' if prediction[0] == 1 else 'Bad Quality Wine'}")

# def on_load_data_click():
#     global model, scaler, X_test_scaled
#     wine_dataset = load_data()
#     if wine_dataset is not None:
#         X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(wine_dataset)
#         model = train_model(X_train_scaled, y_train)
#         accuracy = evaluate_model(model, X_test_scaled, y_test)
#         messagebox.showinfo("Info", f"Data loaded and model trained successfully.\nAccuracy: {accuracy:.2f}")

# def main():
#     global entry_fields, result_label, model, scaler, X_test_scaled
#     root = tk.Tk()
#     root.title("Wine Quality Prediction")

#     # Add custom styling
#     root.configure(bg='#f0f0f0')

#     entry_fields = []
#     attributes = ['Fixed Acidity', 'Volatile Acidity', 'Citric Acid', 'Residual Sugar', 'Chlorides',
#                   'Free Sulfur Dioxide', 'Total Sulfur Dioxide', 'Density', 'pH', 'Sulphates', 'Alcohol']
#     for i, attribute in enumerate(attributes):
#         label = tk.Label(root, text=attribute, bg='#f0f0f0', fg='#333', font=('Arial', 12, 'bold'))
#         label.grid(row=i, column=0, padx=10, pady=5, sticky='e')
#         entry = tk.Entry(root, width=10, font=('Arial', 12))
#         entry.grid(row=i, column=1, padx=10, pady=5, sticky='w')
#         entry_fields.append(entry)

#     predict_button = tk.Button(root, text="Predict", command=lambda: on_predict_click(X_test_scaled, scaler),
#                                 font=('Arial', 12), bg='#4CAF50', fg='#fff', padx=10)
#     predict_button.grid(row=len(entry_fields), columnspan=2, pady=10)

#     load_button = tk.Button(root, text="Load Data", command=on_load_data_click,
#                              font=('Arial', 12), bg='#007BFF', fg='#fff', padx=10)
#     load_button.grid(row=len(entry_fields)+1, columnspan=2, pady=10)

#     result_label = tk.Label(root, text="", bg='#f0f0f0', font=('Arial', 14, 'bold'))
#     result_label.grid(row=len(entry_fields)+2, columnspan=2, pady=10)

#     root.mainloop()

# if __name__ == "__main__":
#     main()

