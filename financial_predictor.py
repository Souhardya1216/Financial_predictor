import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Function to load the dataset
def load_file(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type! Please upload a CSV or Excel file.")
        return None

# Function to train a new model
def train_new_model(data, features, target):
    X = data[features]
    y = data[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Save the model
    joblib.dump(model, "user_model.pkl")

    return mse

# Streamlit UI
st.title("Hybrid Financial Predictor")
st.write("Upload your dataset for bulk predictions or enter numerical values for a single prediction.")

# Section 1: File Upload
st.write("### Option 1: Upload Dataset")
uploaded_file = st.file_uploader("Upload a CSV or Excel file:", type=["csv", "xlsx"])

if uploaded_file is not None:
    data = load_file(uploaded_file)

    if data is not None:
        st.write("### Uploaded Dataset:")
        st.dataframe(data)

        # Column selection
        st.write("### Select Features and Target:")
        columns = list(data.columns)
        selected_features = st.multiselect("Select Features (independent variables):", columns)
        target_column = st.selectbox("Select Target (dependent variable):", columns)

        if selected_features and target_column:
            st.write(f"Features: {selected_features}")
            st.write(f"Target: {target_column}")

            if st.button("Train Model"):
                mse = train_new_model(data, selected_features, target_column)
                st.success(f"Model trained successfully! Mean Squared Error: {mse:.2f}")

                # Predict new data
                st.write("### Upload New Data for Prediction")
                prediction_file = st.file_uploader("Upload a file for prediction (CSV or Excel):", type=["csv", "xlsx"])
                if prediction_file is not None:
                    prediction_data = load_file(prediction_file)

                    if prediction_data is not None:
                        try:
                            model = joblib.load("user_model.pkl")
                            predictions = model.predict(prediction_data[selected_features])
                            prediction_data["Predicted Target"] = predictions
                            st.write("### Predictions:")
                            st.dataframe(prediction_data)

                            # Download predictions
                            csv = prediction_data.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Predictions as CSV",
                                data=csv,
                                file_name="predictions.csv",
                                mime="text/csv",
                            )
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")

# Section 2: Manual Input
st.write("### Option 2: Enter Numerical Values")

# Step 1: Initialize session state for manual input
if "num_features" not in st.session_state:
    st.session_state.num_features = 3
if "manual_features" not in st.session_state:
    st.session_state.manual_features = [f"Feature_{i + 1}" for i in range(st.session_state.num_features)]
if "feature_values" not in st.session_state:
    st.session_state.feature_values = {name: 0.0 for name in st.session_state.manual_features}

# Step 2: Number of features
num_features = st.number_input(
    "How many features do you want to define?",
    min_value=1,
    max_value=20,
    value=st.session_state.num_features,
    step=1,
    key="num_features_input"
)

# Adjust the manual_features and feature_values when the number of features changes
if num_features != st.session_state.num_features:
    # Update the number of features
    st.session_state.num_features = int(num_features)

    # Resize manual_features list
    if len(st.session_state.manual_features) < st.session_state.num_features:
        # Add new features with default names
        new_features = [f"Feature_{i + 1}" for i in
                        range(len(st.session_state.manual_features), st.session_state.num_features)]
        st.session_state.manual_features.extend(new_features)
    else:
        # Truncate the list if fewer features are needed
        st.session_state.manual_features = st.session_state.manual_features[:st.session_state.num_features]

    # Synchronize feature_values dictionary
    st.session_state.feature_values = {
        name: st.session_state.feature_values.get(name, 0.0)
        for name in st.session_state.manual_features
    }

# Step 3: Input for feature names and values
st.write("### Define Features")
for i, feature_name in enumerate(st.session_state.manual_features):
    # Feature name input
    new_name = st.text_input(
        f"Enter name for Feature {i + 1}:",
        value=feature_name,
        key=f"feature_name_{i}"
    )
    if new_name != feature_name:
        # Update name in session state
        st.session_state.manual_features[i] = new_name

        # Update feature_values with the new name
        st.session_state.feature_values[new_name] = st.session_state.feature_values.pop(feature_name, 0.0)

    # Feature value input
    st.session_state.feature_values[new_name] = st.number_input(
        f"Enter value for {new_name}:",
        value=st.session_state.feature_values[new_name],
        key=f"feature_value_{i}"
    )

# Step 4: Prediction Button
if st.button("Predict"):
    try:
        # Load the saved model
        model = joblib.load("user_model.pkl")

        # Prepare input data
        input_data = pd.DataFrame([st.session_state.feature_values.values()], columns=st.session_state.manual_features)

        # Perform prediction
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Target Value: {prediction:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")







