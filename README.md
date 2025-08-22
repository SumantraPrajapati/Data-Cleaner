
#  Data Cleaner

Data Cleaner is a lightweight Python-based tool designed to clean and preprocess messy datasets.  
It helps in handling missing values, removing duplicates, normalizing data, and making datasets ready for analysis or machine learning projects.

---

##  Features
- Remove duplicate rows
- Handle missing values (drop or fill)
- Normalize/standardize data
- Drop unwanted columns
- Save cleaned dataset into a new CSV file

---
## Project Structure
```
DataCleaner/
│── cleaner.py         # Main cleaning script
│── requirements.txt   # Python dependencies
│── sample.csv         # Example dataset (for testing)
```

---

## Requirements
Make sure you have Python installed (>=3.8).  
Install the required libraries:

```bash
pip install -r requirements.txt
```
---
## How to Run
- Clone the repository

```bash
git clone https://github.com/SumantraPrajapati/Data-Cleaner.git
cd DataCleaner
```
- Dependencies installation
  
```bash
pip install -r requirements.txt
```

- Running the Script
```bash
python main.py
```
---

# Convert to EXE
If you want to create a standalone ```.exe ``` file:

- Install PyInstaller
```bash
pip install pyinstaller
```
- Run the following command:
```bash
pyinstaller -onefile main.py
```
- After the build, your ```.exe``` will be located inside the ```dist/``` folder.

---
