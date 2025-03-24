import os
from email.message import EmailMessage

# Email details
sender_email = "your_email@example.com"
receiver_email = "recipient@example.com"
subject = "Facility Lender Share Adjustment Notification"
body = """Dear Recipient,

Please find attached the latest facility lender share adjustment details.

Best regards,  
ROBERTS SCOTT  
XYZ Bank N.A.  
"""

# Create email message
msg = EmailMessage()
msg["From"] = sender_email
msg["To"] = receiver_email
msg["Subject"] = subject
msg.set_content(body)

# Attach the file 1
# file_path = "/Users/karthikeyansethuraman/SourceControl/learning/hackathon-2025/Quantum-coder-ai/python-AI/input/sample2.txt"
# with open(file_path, "rb") as f:
#     file_data = f.read()
#     file_name = os.path.basename(file_path)

# msg.add_attachment(file_data, maintype="text", subtype="plain", filename=file_name)

# Attach the file 2
# file_path = "/Users/karthikeyansethuraman/SourceControl/learning/hackathon-2025/Quantum-coder-ai/python-AI/input/sample3.txt"
# with open(file_path, "rb") as f:
#     file_data = f.read()
#     file_name = os.path.basename(file_path)

# msg.add_attachment(file_data, maintype="text", subtype="plain", filename=file_name)

# Attach the file image
file_path = "/Users/karthikeyansethuraman/SourceControl/learning/hackathon-2025/Quantum-coder-ai/python-AI/input/NOC.png"
with open(file_path, "rb") as f:
    file_data = f.read()
    file_name = os.path.basename(file_path)

msg.add_attachment(file_data, maintype="text", subtype="plain", filename=file_name)

# Save as .eml file
eml_path = "/Users/karthikeyansethuraman/SourceControl/learning/hackathon-2025/Quantum-coder-ai/python-AI/output/sample3.eml"
with open(eml_path, "wb") as eml_file:
    eml_file.write(msg.as_bytes())

print(f"EML file created: {eml_path}")