import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
st.title("Income Prediction App")
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with `age` and `experiance` and `income` columns", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Title
    st.header("Dataset Preview")
    st.dataframe(df.head())

    # Initialize session state variables if they don't exist
    if 'lr_model' not in st.session_state:
        st.session_state.lr_model = LinearRegression()

    if 'data' not in st.session_state:
        st.session_state.data = None

    if 'target' not in st.session_state:
        st.session_state.target = None

    # Display dataset scatter plot
    st.subheader("Original Data: age, experience vs income")
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x = df["age"]
    y = df["experience"]
    z = df["income"]

    # Scatter plot
    ax.scatter(x, y, z, color='green')

    # Adding titles and labels
    ax.set_title('Income earned with respect to Age and Experience')
    ax.set_xlabel('Age(year)')
    ax.set_ylabel('Experience(year)')
    ax.set_zlabel('Income(Rs)')

    # Set a custom view (elevation and azimuthal angle)
    ax.view_init(elev=15, azim=45)

    st.pyplot(fig)

    # Buttons for model training and showing results
    if st.button("Train Model"):
        # Train the model
        st.session_state.data = df[["age", "experience"]].values
        st.session_state.target = df["income"].values
        st.session_state.lr_model.fit(st.session_state.data, st.session_state.target)
        st.success("Model trained successfully!")

    if st.button("Show Training Results"):
        if st.session_state.data is not None and st.session_state.target is not None:
            # Generate predictions and calculate R²
            y_pred = st.session_state.lr_model.predict(st.session_state.data)
            r2 = r2_score(st.session_state.target, y_pred)

            # Display R² score
            st.subheader(f"R² Score: {r2:.2f}")

            # Plot original vs predictions
            st.subheader("Predictions vs Actual")
            fig = plt.figure(figsize=(15, 6))

            # 3D Scatter Plot for Actual Data
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(df["age"], df["experience"], df["income"], color="red", label="Actual Prices")
            ax1.set_title('Income earned with respect to Age and Experience')
            ax1.set_xlabel('Age(year)')
            ax1.set_ylabel('Experience(year)')
            ax1.set_zlabel('Income(Rs)')
            ax1.view_init(elev=15, azim=45)
            ax1.legend()

            # 3D Scatter Plot for Predicted Data
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.scatter(df["age"], df["experience"], df["income"], color="blue", label="Predicted Prices")
            ax2.set_title('Income earned with respect to Age and Experience')
            ax2.set_xlabel('Age(year)')
            ax2.set_ylabel('Experience(year)')
            ax2.set_zlabel('Income(Rs)')
            ax2.view_init(elev=15, azim=45)
            ax2.legend()

            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Please train the model first.")

    # Custom prediction form
    st.subheader("Predict the person's income")
    custom_age = st.number_input("Enter the age (in year):", min_value=0, value=None, step=1)
    exp_age = st.number_input("Enter the experience (in year):", min_value=0, value=None, step=1)
    if st.button("Predict Income"):
        if not hasattr(st.session_state.lr_model, "coef_"):
            st.warning("Please train the model first!")
        else:
            predicted_price = st.session_state.lr_model.predict([[custom_age, exp_age]])[0]
            st.write(f"The predicted income for a person with age: {custom_age} and experience: {exp_age}years is Rs{predicted_price:,.2f}")

else:
    st.warning("Please upload a CSV file to proceed!")
