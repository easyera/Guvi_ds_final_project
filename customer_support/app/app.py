import sys
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / 'src'))

from pipelines.predict_pipeline import predict

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Customer Support Ticket Analyzer", layout="centered")
st.title("Customer Support Ticket Analyzer")
st.markdown("Fill in the ticket details below and click **Predict** to get predictions.")

# ── Input form ─────────────────────────────────────────────────────────────────
with st.form("ticket_form"):

    st.subheader("Ticket Information")
    issue_description = st.text_area("Issue Description", height=100)
    product           = st.selectbox("Product", [
        "billing system", "api service", "analytics dashboard", "cloud storage",
        "crm platform", "ecommerce store", "mobile app", "payment gateway",
        "subscription service", "web portal"
    ])
    category          = st.selectbox("Category", [
        "account suspension", "bug report", "data sync issue", "feature request",
        "login issue", "payment problem", "performance issue", "refund request",
        "security concern", "subscription cancellation"
    ])
    channel           = st.selectbox("Channel", ["chat", "email", "phone", "social media", "web form"])
    region            = st.selectbox("Region", ["africa", "asia", "australia", "europe", "north america", "south america"])

    st.subheader("Customer Information")
    col1, col2 = st.columns(2)
    with col1:
        customer_age            = st.number_input("Customer Age", min_value=18, max_value=75, value=30)
        customer_tenure_months  = st.number_input("Tenure (Months)", min_value=1, max_value=60, value=12)
        previous_tickets        = st.number_input("Previous Tickets", min_value=0, max_value=20, value=3)
        first_response_time_hours = st.number_input("First Response Time (Hours)", min_value=0.0, value=2.0)
    with col2:
        customer_gender    = st.selectbox("Gender", ["female", "male", "other"])
        subscription_type  = st.selectbox("Subscription", ["basic", "enterprise", "free", "premium"])
        customer_segment   = st.selectbox("Segment", ["corporate", "individual", "small business"])
        escalated          = st.selectbox("Escalated", ["no", "yes"])

    st.subheader("Technical Information")
    col3, col4 = st.columns(2)
    with col3:
        operating_system = st.selectbox("OS", ["android", "ios", "linux", "macos", "windows"])
        browser          = st.selectbox("Browser", ["chrome", "edge", "firefox", "safari", "unknown"])
    with col4:
        payment_method        = st.selectbox("Payment Method", ["bank transfer", "credit card", "crypto", "debit card", "paypal"])
        language              = st.selectbox("Language", ["chinese", "english", "french", "german", "japanese", "spanish"])

    preferred_contact_time = st.selectbox("Preferred Contact Time", ["afternoon", "evening", "morning", "night"])
    ticket_created_date    = st.date_input("Ticket Created Date")

    submitted = st.form_submit_button("Predict", use_container_width=True)

# ── Predictions ────────────────────────────────────────────────────────────────
if submitted:
    if not issue_description.strip():
        st.error("Please enter an issue description.")
    else:
        ticket = {
            'issue_description':        issue_description,
            'product':                  product,
            'category':                 category,
            'channel':                  channel,
            'region':                   region,
            'customer_age':             customer_age,
            'customer_gender':          customer_gender,
            'subscription_type':        subscription_type,
            'customer_tenure_months':   customer_tenure_months,
            'previous_tickets':         previous_tickets,
            'first_response_time_hours': first_response_time_hours,
            'escalated':                escalated,
            'operating_system':         operating_system,
            'browser':                  browser,
            'payment_method':           payment_method,
            'language':                 language,
            'preferred_contact_time':   preferred_contact_time,
            'customer_segment':         customer_segment,
            'ticket_created_date':      str(ticket_created_date),
            # These are passed to avoid clean_data errors — not used for prediction
            'ticket_resolved_date':     str(ticket_created_date),
            'status':                   'open',
            'sla_breached':             'no',
            'customer_satisfaction_score': 3,
            'resolution_notes':         '',
            'issue_complexity_score':   3,
            'resolution_time_hours':    0.0
        }

        with st.spinner("Predicting..."):
            result = predict(ticket)

        st.success("Predictions Complete!")
        st.subheader("Results")

        col5, col6, col7 = st.columns(3)
        with col5:
            st.metric("Priority", result['priority'].upper())
        with col6:
            st.metric("SLA Breach", result['sla_breach'])
        with col7:
            st.metric("Satisfaction Score", f"{result['satisfaction']} / 5")