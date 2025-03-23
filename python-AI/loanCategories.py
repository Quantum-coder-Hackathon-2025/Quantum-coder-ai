import pandas as pd

# Define Categories and Subcategories
categories = {
    "Account Maintenance": ["Update Account Information", "Change Payment Schedule", "Interest Rate Adjustment", "Loan Restructuring"],
    "Payment Processing": ["Scheduled Payment", "Prepayment", "Partial Payment", "Late Payment", "Payment Reversal"],
    "Fee Management": ["Fee Payment", "Ongoing Fee", "Amendment Fee", "Reallocation Fee", "Letter of Credit Fee"],
    "Money Movement - Inbound": ["Principal", "Interest", "Principal + Interest", "Principal + Interest + Fee"],
    "Money Movement - Outbound": ["Timebound", "Foreign Currency"],
    "Adjustments": ["Loan Adjustment", "AU Transfer"],
    "Commitment Changes": ["Cashless Roll", "Increase Commitment", "Decrease Commitment"],
    "Closing Requests": ["Closing Notice", "Reallocation Principal"],
    "Collateral Management": ["Collateral Valuation", "Collateral Substitution", "Collateral Release", "Collateral Reassessment"],
    "Customer Service Requests": ["Inquiry Handling", "Dispute Resolution", "Statement Generation", "Account Information Requests"],
    "Compliance and Reporting": ["Regulatory Compliance", "Internal Audits", "External Audits", "Compliance Reporting"],
    "Risk Management": ["Credit Risk Assessment", "Market Risk Assessment", "Operational Risk Assessment", "Risk Mitigation Strategies"],
    "Loan Monitoring": ["Delinquency Monitoring", "Covenant Monitoring", "Financial Performance Monitoring", "Early Warning Systems"],
    "Collections and Recovery": ["Delinquency Notices", "Collections Calls", "Legal Actions", "Recovery Strategies"],
    "Loan Modifications": ["Forbearance Agreements", "Loan Extensions", "Interest Rate Modifications", "Principal Reductions"],
    "Escrow Management": ["Tax Payments", "Insurance Payments", "Escrow Analysis", "Escrow Disbursements"],
    "Documentation Management": ["Document Storage", "Document Retrieval", "Document Updates", "Document Compliance"],
    "Customer Communication": ["Notification Management", "Email Communication", "SMS Alerts", "Customer Portals"],
    "Financial Reporting": ["Loan Performance Reports", "Financial Statements", "Budgeting and Forecasting", "Profit and Loss Analysis"],
    "Technology and Automation": ["Loan Servicing Software", "Automated Payment Systems", "Digital Communication Channels", "Data Analytics and Reporting Tools"],
    "Training and Development": ["Staff Training Programs", "Compliance Training", "Customer Service Training", "Technology Training"]
}

# Convert to a DataFrame
data = []
for category, subcategories in categories.items():
    for subcategory in subcategories:
        data.append([category, subcategory])

df = pd.DataFrame(data, columns=["Category", "Subcategory"])

# Save to Excel
file_path = "loan_categories.xlsx"
df.to_excel(file_path, index=False)

print(f"Excel file created: {file_path}")