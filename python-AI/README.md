# Quantum-coder-ai
created repo for hackathon event
Create a New Python Project
Open VS Code and create a new folder for your project
Open the folder in VS Code (File > Open Folder)
Inside the folder, create a new file with a .py extension (e.g., main.py)
1. Install VS Code and Python
Download and install Visual Studio Code
Install Python and make sure it's added to the system path
2. Install Python Extension in VS Code
Open VS Code
Go to Extensions (Ctrl+Shift+X), search for Python, and install the official Microsoft extension

3. Create and Activate a Virtual Environment (Optional but Recommended)
    Open the terminal in VS Code (Ctrl+ `) and run:
    python -m venv venv  # Creates a virtual environment

Activate the virtual environment:
    Windows (Command Prompt):
    windows -> venv\Scripts\activate
    mac -> source venv/bin/activate
4. install all library
    pip install -requirements.txt
    or do one by one installation
    pip install 
        pandas
        fastapi
        uvicorn
        scikit-learn
        numpy
        pdfplumber
        python-docx
        python-multipart
        spacy
        joblib

python -m spacy download en_core_web_sm

5. Run the application
    uvicorn emailClassifier:app --reload