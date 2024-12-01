# Income Prediction

A Streamlit-based application for predicting income based on the age and experience of a person.

Access the app through: https://incomepredictor-wizsvszjazdhdyjmgp32rr.streamlit.app/


## Features

- Upload a CSV file containing age, experience and income data.
- Visualize the uploaded data and its graph.
- Train a model using the uploaded dataset.
- View training results, including R² score and graphs of actual vs. predicted incomes.
- Perform live predictions by inputting the age and experience of a person.


## Requirements

- Python (latest version)
- Streamlit package


## Installation

1. Install the latest version of Python from [python.org](https://www.python.org/).
2. Open a command prompt and install Streamlit:
   ```bash
   pip install streamlit
  

## How To Run
1. Open a command prompt and navigate to the directory containing app.py
2. Run the following command:
   ```bash
   streamlit run <path-to-file>\app.py

4. The app will open in your web browser.


## Usage
### 1. Upload Data:
1. Upload a CSV file with age, experience and income.
2. Preview the data and view its graph.

### 2. Train the Model:
1. Click the Train Model button.
2. A success message ("Model trained successfully!") will be displayed.

### 3. View Training Results:
1. Click the Show Training Results button.
2. View the R² score and a graph of actual vs. predicted prices.

### 4. Live Prediction:
1. Input the age and experience of a person.
2. View the predicted income instantly.


## Example CSV file
```bash
age,experience,income
25,1,30450
30,3,35670
47,2,31580
```


## Output
1. R² Score of the trained model.
2. Graph of actual vs. predicted incomes.
3. Real-time income predictions.





