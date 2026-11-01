import os

import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine,text

DATABASE_URL=os.getenv(
    "DATABASE_URL",
    "sqlite:///./artifacts/generation_quality_gate.db",
)
API_URL = os.getenv("API_URL", "http://localhost:8000")

engine=create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

st.set_page_config(page_title="Generation Quality Gate",layout="wide",)

st.title("Generation Quality Gate")
st.caption("Model serving and quality monitoring")

with st.expander("Score a Shot"):
    request_id = st.text_input("Request ID", value="dashboard-request")
    project_id = st.text_input("Project ID", value="dashboard-project")
    features_json = st.text_area("Features JSON", value="{}")
    if st.button("Score"):
        try:
            import json
            response = requests.post(f"{API_URL}/v1/score", json={"request_id": request_id, "project_id": project_id, "features": json.loads(features_json)}, timeout=30)
            response.raise_for_status()
            st.json(response.json())
        except Exception as error:
            st.error(f"Scoring failed: {error}")

def query_dataframe(query):
    with engine.connect() as connection:
        return pd.read_sql(text(query),connection)

try:
    scores=query_dataframe(
        """
        SELECT
            request_id,
            project_id,
            model_version,
            identity_score,
            defect_score,
            adherence_score,
            decision,
            created_at
        FROM scores
        ORDER BY created_at DESC
        """
    )

    feedback=query_dataframe(
        """
        SELECT
            feedback,
            created_at
        FROM feedback
        ORDER BY created_at DESC
        """
    )

    rollout=query_dataframe(
        """
        SELECT
            model_version,
            stage,
            canary_percentage,
            updated_at
        FROM rollout_state
        ORDER BY updated_at DESC
        LIMIT 1
        """
    )

    col1,col2,col3,col4=st.columns(4)

    col1.metric("Total Scores",len(scores))

    if not scores.empty:
        col2.metric("KEEP",int((scores["decision"]=="KEEP").sum()))
        col3.metric("REGENERATE",int((scores["decision"]=="REGENERATE").sum()))
        col4.metric("REVIEW",
            int((scores["decision"]=="REVIEW").sum()))
    else:
        col2.metric("KEEP",0)
        col3.metric("REGENERATE",0)
        col4.metric("REVIEW",0)

    st.subheader("Decision Distribution")
    if not scores.empty:
        decision_counts=(scores["decision"].value_counts().rename_axis("decision").reset_index(name="count"))
        st.bar_chart(decision_counts.set_index("decision"))
    else:
        st.info("No scoring data available yet.")

    st.subheader("Recent Scores")

    if not scores.empty:
        st.dataframe(
            scores.head(100),
            use_container_width=True,
        )
    else:
        st.info("No scores available.")

    st.subheader("Feedback")

    if not feedback.empty:
        feedback_counts=(feedback["feedback"].value_counts().rename_axis("feedback").reset_index(name="count"))

        st.bar_chart(feedback_counts.set_index("feedback"))

        st.dataframe(feedback.head(100),use_container_width=True)
    else:
        st.info("No feedback available.")

    st.subheader("Rollout")

    if not rollout.empty:
        st.dataframe(
            rollout,
            use_container_width=True,
        )
    else:
        st.info("No rollout state available.")

    st.subheader("Model Distribution")

    if not scores.empty:
        model_counts=(scores["model_version"].value_counts().rename_axis("model_version").reset_index(name="requests"))

        st.bar_chart(
            model_counts.set_index("model_version")
        )
    else:
        st.info("No model usage data available.")

except Exception as error:
    st.error(f"Unable to connect to PostgreSQL: {error}")
