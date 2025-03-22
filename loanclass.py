from collections import defaultdict

def classify_loan_request(request):
    categories = {
        "Account Maintenance": ["update account", "change payment schedule", "interest rate adjustment", "loan restructuring"],
        "Payment Processing": ["scheduled payment", "prepayment", "partial payment", "late payment", "payment reversal"],
        "Fee Management": ["fee payment", "ongoing fee", "amendment fee", "reallocation fee", "letter of credit fee"],
        "Money Movement": ["inbound money movement", "outbound money movement", "principal", "interest", "foreign currency"],
        "Loan Adjustments": ["adjustment", "au transfer"],
        "Commitment Changes": ["cashless roll", "increase commitment", "decrease commitment"],
        "Closing Requests": ["closing notice", "reallocation principal"],
        "Collateral Management": ["collateral valuation", "collateral substitution", "collateral release", "collateral reassessment"],
        "Customer Service Requests": ["inquiry handling", "dispute resolution", "statement generation", "account information request"],
        "Compliance and Reporting": ["regulatory compliance", "internal audits", "external audits", "compliance reporting"],
        "Risk Management": ["credit risk assessment", "market risk assessment", "operational risk assessment", "risk mitigation strategies"],
        "Loan Monitoring": ["delinquency monitoring", "covenant monitoring", "financial performance monitoring", "early warning systems"],
        "Collections and Recovery": ["delinquency notices", "collections calls", "legal actions", "recovery strategies"],
        "Loan Modifications": ["forbearance agreements", "loan extensions", "interest rate modifications", "principal reductions"],
        "Escrow Management": ["tax payments", "insurance payments", "escrow analysis", "escrow disbursements"],
        "Documentation Management": ["document storage", "document retrieval", "document updates", "document compliance"],
        "Customer Communication": ["notification management", "email communication", "sms alerts", "customer portals"],
        "Financial Reporting": ["loan performance reports", "financial statements", "budgeting and forecasting", "profit and loss analysis"],
        "Technology and Automation": ["loan servicing software", "automated payment systems", "digital communication channels", "data analytics and reporting tools"],
        "Training and Development": ["staff training programs", "compliance training", "customer service training", "technology training"]
    }
    
    request_lower = request.lower()
    
    matches = defaultdict(list)
    
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in request_lower:
                matches[category].append(keyword)
    
    if matches:
        return {"Category": list(matches.keys()), "Matched Keywords": dict(matches)}
    else:
        return {"Category": "Uncategorized", "Matched Keywords": []}

# Example Usage
request_input = "I need to change my payment schedule and update my account information."
result = classify_loan_request(request_input)
print(result)
