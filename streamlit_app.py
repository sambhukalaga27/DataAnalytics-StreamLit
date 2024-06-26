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

def correlation_plot(data, features, target):
    if not isinstance(features, list):
        features = list(features)

    selected_columns = features + [target]
    data_filtered = data[selected_columns]
    corr_matrix = data_filtered.corr()

    fig = px.imshow(corr_matrix,
                    labels=dict(x="Feature", y="Feature", color="Correlation"),
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    title="Correlation matrix between target and top features",
                    aspect="auto")
    st.plotly_chart(fig, use_container_width=True)
    return corr_matrix

def generate_description(correlations):
    descriptions = []
    for index, value in correlations.items():
        if value > 0.5:
            description = f"Strong positive correlation: As {index} increases, the target significantly increases."
        elif value > 0:
            description = f"Positive correlation: As {index} increases, the target tends to increase."
        elif value < -0.5:
            description = f"Strong negative correlation: As {index} increases, the target significantly decreases."
        elif value < 0:
            description = f"Negative correlation: As {index} increases, the target tends to decrease."
        else:
            description = f"Weak correlation: {index} has little to no direct effect on the target."
        descriptions.append(description)
    return descriptions

def display_correlation_and_descriptions(data, features, target_feature):
    corr_matrix = correlation_plot(data, features, target_feature)
    corr_values = corr_matrix[target_feature].drop(target_feature).to_frame()
    corr_values.columns = ['Correlation with Target']
    descriptions = generate_description(corr_values['Correlation with Target'])
    corr_values['Description'] = descriptions
    st.write(f"Correlation with Target Feature: {target_feature}")
    st.dataframe(corr_values)

def plot_assisted_runs(data, selected_pphid, overall_avg_ratio):
    if selected_pphid != "All":
        data = data[data['pphid'] == selected_pphid]

    # data['DayN'] = pd.to_datetime(data['DayN'])
    data_grouped = data.groupby(['DayN', 'assisted']).size().unstack(fill_value=0)

    fig, ax = plt.subplots()
    for label in data_grouped.columns:
        ax.plot(data_grouped.index, data_grouped[label], marker='o', linestyle='--', label=f'Assisted={label}')

    ax.set_xlabel('Run Day')
    ax.set_ylabel('Count')
    ax.set_title(f'Assisted vs Unassisted Runs for {selected_pphid}')
    ax.legend()
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    st.pyplot(fig)

    unassisted_counts = data_grouped.get(0, pd.Series(0, index=data_grouped.index))
    assisted_counts = data_grouped.get(1, pd.Series(0, index=data_grouped.index))
    avg_ratio_unassisted_assisted = (unassisted_counts / assisted_counts).replace([np.inf, -np.inf], np.nan).mean()

    # Calculate the proportion of assisted to unassisted at each RunDate
    proportions = (assisted_counts / unassisted_counts).replace([np.inf, -np.inf], np.nan).dropna()
    st.write("Proportion of Assisted to Unassisted Runs at each RunDate:")
    st.write(proportions)

    if selected_pphid != "All":
        st.write(f"Average Ratio of Unassisted by Assisted Runs for {selected_pphid}: {avg_ratio_unassisted_assisted:.2f}")

    st.write(f"Average Overall Ratio of Unassisted by Assisted Runs for All Participants: {overall_avg_ratio:.2f}")

def calculate_overall_avg_ratio(data):
    overall_unassisted_counts = data['assisted'].value_counts().get(0, 0)
    st.write(f"Overall Unassisted Counts: {overall_unassisted_counts}")
    overall_assisted_counts = data['assisted'].value_counts().get(1, 0)
    st.write(f"Overall Assisted Counts: {overall_assisted_counts}")
    overall_avg_ratio_unassisted_assisted = overall_unassisted_counts / overall_assisted_counts
    return overall_avg_ratio_unassisted_assisted

def page_with_both_groups():
    st.subheader('Data Analysis and Model Training - Total Cohort Data')

    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        pphids = data['pphid'].unique().tolist()
        pphids.insert(0, "All")
        selected_pphid = st.selectbox("Select Participant ID", pphids)

        overall_avg_ratio = calculate_overall_avg_ratio(data)
        plot_assisted_runs(data, selected_pphid, overall_avg_ratio)

        dropping_features = [
            'sessionID', 'pphid', 'RunDate', 'RunTime', 'RunDateTime', 'surgtype','Success', 'weekday', 'timeOfDay',
            'step1Time','step2Time',"step3Time",'step4Time','step5Time','step6Time',
            "minAccuracy", 'meanAccuracy', 'sessionDateTime_2', 
            'SubSuccess', "step1Accuracy_overall_difficulty","step2Accuracy_overall_difficulty",
            "step3Accuracy_overall_difficulty","step4Accuracy_overall_difficulty","step5Accuracy_overall_difficulty",
            "step6Accuracy_overall_difficulty","step1Accuracy_pphid_difficulty","step2Accuracy_pphid_difficulty",
            "step3Accuracy_pphid_difficulty","step4Accuracy_pphid_difficulty","step5Accuracy_pphid_difficulty",
            "step6Accuracy_pphid_difficulty","maxAccuracy","StepCompletionRate","step1TimeInverse","step2TimeInverse",
            "step3TimeInverse","step4TimeInverse","step5TimeInverse","step6TimeInverse","step1WeightedAccuracy",
            "step2WeightedAccuracy","step3WeightedAccuracy","step4WeightedAccuracy","step5WeightedAccuracy",
            "step6WeightedAccuracy",
            "overallAccuracy",
            "stepnum1","stepnum2","stepnum3","stepnum4","stepnum5","stepnum6"
        ]
        data.drop(columns=dropping_features, inplace=True, errors='ignore')

        features_to_drop = st.multiselect('Select extra features to drop', data.columns)
        data.drop(columns=features_to_drop, inplace=True, errors='ignore')

        if 'pphid' in data.columns:
            data.drop(columns='pphid', inplace=True, errors='ignore')
        
        target_feature = st.selectbox("Select the target feature", data.columns)

        y = data[target_feature].dropna()
        X = data.drop(columns=[target_feature]).loc[y.index]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        st.write('Mean Squared Error:', mse)

        feature_importances = pd.Series(pipeline.named_steps['regressor'].feature_importances_, index=X.columns)
        top_features = feature_importances.nlargest(10)
        top_feature = feature_importances.nlargest(10).index.tolist()

        fig, ax = plt.subplots()
        top_features.plot(kind='barh', ax=ax)
        ax.set_title('Top 10 Feature Importances')
        st.pyplot(fig)

        display_correlation_and_descriptions(data, top_feature, target_feature)

        fig = px.pie(values=top_features, names=top_features.index, title='Top 10 Feature Importances')
        st.plotly_chart(fig, use_container_width=True)

        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=predictions, ax=ax)
        ax.set_xlabel('Actual Scores')
        ax.set_ylabel('Predicted Scores')
        ax.set_title('Predicted vs. Actual Scores')
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=1)
        st.pyplot(fig)

def page_individual_groups():
    st.subheader('Data Analysis and Model Training - Group Based')

    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        
        group = st.sidebar.radio("Select Group", [1, 2])
        data = data[data['Group'] == group]

        pphids = data['pphid'].unique().tolist()
        pphids.insert(0, "All")
        selected_pphid = st.selectbox("Select Participant ID", pphids)

        overall_avg_ratio = calculate_overall_avg_ratio(data)
        plot_assisted_runs(data, selected_pphid, overall_avg_ratio)

        dropping_features = [
            'sessionID', 'pphid', 'RunDate', 'RunTime', 'RunDateTime', 'surgtype','Success', 'weekday', 'timeOfDay',
            'step1Time','step2Time',"step3Time",'step4Time','step5Time','step6Time',"minAccuracy",
            'meanAccuracy', 'sessionDateTime_2', 'SubSuccess',"step1Accuracy_overall_difficulty","step2Accuracy_overall_difficulty",
            "step3Accuracy_overall_difficulty","step4Accuracy_overall_difficulty","step5Accuracy_overall_difficulty",
            "step6Accuracy_overall_difficulty","step1Accuracy_pphid_difficulty","step2Accuracy_pphid_difficulty",
            "step3Accuracy_pphid_difficulty","step4Accuracy_pphid_difficulty","step5Accuracy_pphid_difficulty",
            "step6Accuracy_pphid_difficulty","maxAccuracy","StepCompletionRate","step1TimeInverse","step2TimeInverse",
            "step3TimeInverse","step4TimeInverse","step5TimeInverse","step6TimeInverse","step1WeightedAccuracy",
            "step2WeightedAccuracy","step3WeightedAccuracy","step4WeightedAccuracy","step5WeightedAccuracy",
            "step6WeightedAccuracy","overallAccuracy","stepnum1","stepnum2","stepnum3","stepnum4","stepnum5","stepnum6"
        ]
        data.drop(columns=dropping_features, inplace=True, errors='ignore')

        features_to_drop = st.multiselect('Select extra features to drop', data.columns)
        data.drop(columns=features_to_drop, inplace=True, errors='ignore')

        if 'pphid' in data.columns:
            data.drop(columns='pphid', inplace=True, errors='ignore')
        
        target_feature = st.selectbox("Select the target feature", data.columns)

        y = data[target_feature].dropna()
        X = data.drop(columns=[target_feature]).loc[y.index]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        st.write('Mean Squared Error:', mse)

        feature_importances = pd.Series(pipeline.named_steps['regressor'].feature_importances_, index=X.columns)
        top_features = feature_importances.nlargest(10)
        top_feature = feature_importances.nlargest(10).index.tolist()

        fig, ax = plt.subplots()
        top_features.plot(kind='barh', ax=ax)
        ax.setTitle('Top 10 Feature Importances')
        st.pyplot(fig)

        display_correlation_and_descriptions(data, top_feature, target_feature)

        fig = px.pie(values=top_features, names=top_features.index, title='Top 10 Feature Importances')
        st.plotly_chart(fig, use_container_width=True)

        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=predictions, ax=ax)
        ax.set_xlabel('Actual Scores')
        ax.set_ylabel('Predicted Scores')
        ax.set_title('Predicted vs. Actual Scores')
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=1)
        st.pyplot(fig)

def main():
    st.title('Data Analysis and Model Training')
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ["With Both the groups", "Individually group wise"])

    if page == "With Both the groups":
        page_with_both_groups()
    elif page == "Individually group wise":
        page_individual_groups()

if __name__ == "__main__":
    main()
