import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import plotly.express as px


def main():
    st.title('Data Analysis and Model Training')

    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        # Dropping default features
        dropping_features = [
            'sessionID', 'sessionDate', 'sessionTime', 'sessionDateTime', 'surgtype', 'Success', 'weekday',
            'step1Time', 'step2Time', 'step3Time', 'step4Time', 'step5Time', 'step6Time', 'step6Accuracy', 'step5Accuracy',
            'step1Accuracy', 'step2Accuracy', 'step3Accuracy', 'step4Accuracy', 'minAccuracy', 'meanAccuracy', 'timeOfDay',
            'sessionDateTime_2', 'step1Accuracy_overall_difficulty', 'step2Accuracy_overall_difficulty',
            'step3Accuracy_overall_difficulty', 'step4Accuracy_overall_difficulty', 'step5Accuracy_overall_difficulty',
            'step6Accuracy_overall_difficulty', 'step1Accuracy_pphid_difficulty', 'step2Accuracy_pphid_difficulty',
            'step3Accuracy_pphid_difficulty', 'step4Accuracy_pphid_difficulty', 'step5Accuracy_pphid_difficulty',
            'step6Accuracy_pphid_difficulty', 'maxAccuracy', 'StepCompletionRate', 'step1TimeInverse', 'step2TimeInverse',
            'step3TimeInverse', 'step4TimeInverse', 'step5TimeInverse', 'step6TimeInverse', 'step1WeightedAccuracy',
            'step2WeightedAccuracy', 'step3WeightedAccuracy', 'step4WeightedAccuracy', 'step5WeightedAccuracy',
            'step6WeightedAccuracy', 'overallAccuracy'
        ]
        data.drop(columns=dropping_features, inplace=True, errors='ignore')

        # Dynamically select additional features to drop
        st.write("Select features to drop:")
        features_to_drop = st.multiselect('Select extra features to drop', data.columns)
        data.drop(columns=features_to_drop, inplace=True, errors='ignore')

        # Selecting target feature
        if 'pphid' in data.columns:
            data.drop(columns='pphid', inplace=True, errors='ignore')
        
        target_feature = st.selectbox("Select the target feature", data.columns)

        # Data preparation
        y = data[target_feature].dropna()
        X = data.drop(columns=[target_feature]).loc[y.index]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        st.write('Mean Squared Error:', mse)

        # Plotting feature importances
        feature_importances = pd.Series(pipeline.named_steps['regressor'].feature_importances_, index=X.columns)
        top_features = feature_importances.nlargest(10)
        
        fig, ax = plt.subplots()
        top_features.plot(kind='barh', ax=ax)
        ax.set_title('Top 10 Feature Importances')
        st.pyplot(fig)

        # Interactive Pie Chart for Top Features using Plotly
        fig = px.pie(values=top_features, names=top_features.index, title='Top 10 Feature Importances')
        st.plotly_chart(fig, use_container_width=True)

        # Predicted vs Actual plot
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=predictions, ax=ax)
        ax.set_xlabel('Actual Scores')
        ax.set_ylabel('Predicted Scores')
        ax.set_title('Predicted vs. Actual Scores')
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=1)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
