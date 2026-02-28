#save as emi_calculator.py
#make sure streamlit is installed (pip install streamlit)
#run using streamlit run > streamlit run emi_calculator.py
#improve the code to integrate AI using free and simple models like 'google-flan-t5-base' or -xl, -large.

import streamlit as st

# EMI calculation function
def calculate_emi(principal, annual_rate, years):
    monthly_rate = annual_rate / 12 / 100
    months = years * 12
    
    if monthly_rate == 0:
        emi = principal / months
    else:
        emi = (principal * monthly_rate * (1 + monthly_rate) ** months) / \
              ((1 + monthly_rate) ** months - 1)
    
    return emi, months

# Streamlit UI
st.title("🏦 Loan EMI Calculator")

st.write("Calculate your monthly EMI easily")

# Inputs
principal = st.number_input("Loan Amount (₹)", min_value=0.0, value=500000.0)
rate = st.number_input("Annual Interest Rate (%)", min_value=0.0, value=10.0)
years = st.number_input("Loan Tenure (Years)", min_value=1, value=5)

# Calculate button
if st.button("Calculate EMI"):
    
    emi, months = calculate_emi(principal, rate, years)
    
    total_payment = emi * months
    total_interest = total_payment - principal
    
    st.success(f"Monthly EMI: ₹ {emi:,.2f}")
    
    st.info(f"""
    Loan Summary:
    
    • Loan Amount: ₹ {principal:,.2f}  
    • Interest Rate: {rate}%  
    • Tenure: {years} years  
    • Total Payment: ₹ {total_payment:,.2f}  
    • Total Interest: ₹ {total_interest:,.2f}  
    """)

# Optional visualization
if st.checkbox("Show Payment Breakdown"):
    
    interest = total_interest if 'total_interest' in locals() else 0
    principal_amount = principal if 'principal' in locals() else 0
    
    st.bar_chart({
        "Amount": [principal_amount, interest]
    })
