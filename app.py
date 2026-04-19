import streamlit as st
import joblib
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import plotly.express as px
from utils import preprocess_text

# Set page configs
st.set_page_config(page_title="AI Task Dashboard", layout="wide", page_icon="🚀")

# ---------------------------------------------------------
# Custom styling removed for Dark/Light mode support
# ---------------------------------------------------------

# ---------------------------------------------------------
# Initialization & Data
# ---------------------------------------------------------
if 'tasks_df' not in st.session_state:
    st.session_state.tasks_df = pd.DataFrame(columns=[
        "Task", "Predicted Priority", "Confidence", "Due Date", "Status", "Timestamp"
    ])

@st.cache_resource
def load_models():
    model_path = 'models/best_model.pkl'
    vec_path = 'models/tfidf_vectorizer.pkl'
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        return None, None
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer

model, vectorizer = load_models()



# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.title("🚀 Navigation")
page = st.sidebar.radio("Go to", ["📊 Dashboard", "➕ Add Task", "📋 Task Manager", "📈 Analytics"])

df = st.session_state.tasks_df

if model is None or vectorizer is None:
    st.error("⚠️ Models not found! Please train the model first by running `python train.py`.")
    st.stop()

# ---------------------------------------------------------
# PAGES
# ---------------------------------------------------------

if page == "📊 Dashboard":
    st.title("📊 Task Dashboard Overview")
    st.markdown("---")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_tasks = len(df)
    critical_count = len(df[df["Predicted Priority"].str.lower() == "critical"]) if total_tasks > 0 else 0
    high_count = len(df[df["Predicted Priority"].str.lower() == "high"]) if total_tasks > 0 else 0
    med_count = len(df[df["Predicted Priority"].str.lower() == "medium"]) if total_tasks > 0 else 0
    low_count = len(df[df["Predicted Priority"].str.lower() == "low"]) if total_tasks > 0 else 0
    
    with col1:
        st.metric("Total Tasks", total_tasks)
    with col2:
        st.metric("Critical Priority", critical_count)
    with col3:
        st.metric("High Priority", high_count)
    with col4:
        st.metric("Medium Priority", med_count)
    with col5:
        st.metric("Low Priority", low_count)
        
    st.markdown("### 🕒 Recent Predictions")
    if total_tasks > 0:
        recent_tasks = df.sort_values(by="Timestamp", ascending=False).head(5)
        st.dataframe(recent_tasks, use_container_width=True, hide_index=True)
    else:
        st.info("No tasks added yet. Go to 'Add Task' to create one!")

elif page == "➕ Add Task":
    st.title("➕ Add New Task")
    st.markdown("---")
    
    with st.container():
        task_desc = st.text_area("Task Description:", placeholder="e.g. The database server is crashing frequently...")
        
        col1, col2 = st.columns(2)
        with col1:
            due_date = st.date_input("Due Date", min_value=datetime.date.today())
        with col2:
            status = st.selectbox("Status", ["Pending", "In Progress", "Completed"])
        
        if st.button("Predict & Add Task", use_container_width=True, type="primary"):
            if task_desc.strip() == "":
                st.error("Please enter a valid task description.")
            else:
                with st.spinner("Analyzing text and running prediction..."):
                    cleaned_text = preprocess_text(task_desc)
                    vec_text = vectorizer.transform([cleaned_text])
                    prediction = model.predict(vec_text)[0]
                    
                    # Calculate Confidence Score
                    confidence = "N/A"
                    if hasattr(model, "predict_proba"):
                        proba_array = model.predict_proba(vec_text)[0]
                        max_proba = max(proba_array) * 100
                        confidence = f"{max_proba:.1f}%"
                    elif hasattr(model, "decision_function"):
                        confidence = "Score calculated" # Fallback if probability not available
                    
                    msg = f"**🔥 Predicted Priority:** {str(prediction).upper()}  \n**Confidence:** {confidence}"
                    p = str(prediction).lower()
                    if p == "critical":
                        st.error(msg)
                    elif p == "high":
                        st.warning(msg)
                    elif p == "medium":
                        st.info(msg)
                    else:
                        st.success(msg)
                    
                    # Add to session state
                    new_task = {
                        "Task": task_desc,
                        "Predicted Priority": prediction,
                        "Confidence": confidence,
                        "Due Date": str(due_date),
                        "Status": status,
                        "Timestamp": datetime.datetime.now()
                    }
                    st.session_state.tasks_df = pd.concat([st.session_state.tasks_df, pd.DataFrame([new_task])], ignore_index=True)
                    st.success("✅ Task successfully added to the Task Manager!")

elif page == "📋 Task Manager":
    st.title("📋 Task Manager")
    st.markdown("---")
    
    if len(df) == 0:
        st.info("No tasks available. Please add tasks first.")
    else:
        # Filters
        st.subheader("🔍 Filters")
        col1, col2 = st.columns(2)
        with col1:
            all_priorities = df["Predicted Priority"].unique().tolist()
            selected_priorities = st.multiselect("Filter by Priority", all_priorities, default=all_priorities)
        with col2:
            all_statuses = df["Status"].unique().tolist()
            selected_statuses = st.multiselect("Filter by Status", all_statuses, default=all_statuses)
            
        filtered_df = df[df["Predicted Priority"].isin(selected_priorities) & df["Status"].isin(selected_statuses)]
        
        st.subheader("📝 Task List")
        # Use data editor to allow user to optionally edit or just view (we will use dataframe and a delete button)
        st.dataframe(filtered_df, use_container_width=True)
        
        # Delete functionality
        st.markdown("---")
        st.markdown("### 🗑️ Delete Task")
        tasks_to_delete = st.selectbox("Select Task to Delete", ["-- Select a task --"] + filtered_df["Task"].tolist())
        if st.button("Delete Task") and tasks_to_delete != "-- Select a task --":
            st.session_state.tasks_df = st.session_state.tasks_df[st.session_state.tasks_df["Task"] != tasks_to_delete]
            st.success("Task deleted successfully! Please refresh or switch pages to see changes.")
            st.rerun()
            
        # Optional: CSV Download
        st.markdown("---")
        st.markdown("### 💾 Export")
        csv = st.session_state.tasks_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name='task_manager_data.csv',
            mime='text/csv',
        )

elif page == "📈 Analytics":
    st.title("📈 Analytics")
    st.markdown("---")
    
    if len(df) == 0:
        st.info("Not enough data to calculate analytics. Please add tasks first.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Priority Distribution")
            priority_counts = df["Predicted Priority"].value_counts().reset_index()
            priority_counts.columns = ['Priority', 'Count']
            
            # Map professional aesthetic colors dynamically
            color_map = {
                "Critical": "#EF4444", "High": "#F97316", 
                "Medium": "#3B82F6", "Low": "#10B981"
            }
            # Fallback for lowercase mapped from prediction array
            color_map.update({k.lower(): v for k, v in color_map.items()})

            fig1 = px.pie(
                priority_counts, names='Priority', values='Count', 
                hole=0.45, color='Priority', color_discrete_map=color_map
            )
            fig1.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#000000', width=0)))
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True, theme="streamlit")
            
        with col2:
            st.markdown("### Task Status Counts")
            status_counts = df["Status"].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            
            status_color_map = {
                "Pending": "#F59E0B", "In Progress": "#3B82F6", "Completed": "#10B981"
            }
            
            fig2 = px.bar(
                status_counts, x='Status', y='Count', text_auto=True,
                color='Status', color_discrete_map=status_color_map
            )
            fig2.update_traces(textfont_size=14, textangle=0, textposition="outside", cliponaxis=False)
            fig2.update_layout(xaxis_title="", yaxis_title="Number of Tasks", showlegend=False)
            st.plotly_chart(fig2, use_container_width=True, theme="streamlit")
