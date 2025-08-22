import pandas as pd
import numpy as np
import os
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class AutoDataCleaner:
    def __init__(self):
        self.df = None
        self.file_name = None
        self.report = []

    def find_csv_file(self):
        files = [f for f in os.listdir('.') if f.lower().endswith('.csv')]
        if files:
            self.file_name = files[0]
            print(f"Found CSV file: {self.file_name}")
            return self.file_name
        else:
            print("No CSV file found in the current directory.")
            return None

    def load_dataset(self):
        if self.file_name:
            try:
                self.df = pd.read_csv(self.file_name)
                print(f"Dataset loaded successfully with shape {self.df.shape}")
            except Exception as e:
                print(f"Error loading dataset: {e}")

    def preview(self, rows=5):
        if self.df is not None:
            return self.df.head(rows)
        return "No dataset loaded."

    def analyse_missing_values(self):
        if self.df is None:
            print("No dataset loaded.")
            return

        missing_summary = self.df.isnull().sum()
        missing_percent = (missing_summary / len(self.df)) * 100
        print("\n Missing Values Summary (%):\n", missing_percent)

        # Drop rows with <5% missing
        row_missing = self.df.isnull().mean(axis=1)
        if (row_missing > 0).sum() / len(self.df) < 0.05:
            before = len(self.df)
            self.df.dropna(inplace=True)
            print(f"Dropped {before - len(self.df)} rows with few missing values.")
            self.report.append(f"Dropped {before - len(self.df)} rows with few missing values")
            

        # Drop columns with >50% missing
        high_missing_cols = missing_percent[missing_percent > 50].index
        if len(high_missing_cols) > 0:
            self.df.drop(columns=high_missing_cols, inplace=True)
            print(f"Dropped columns with >50% missing: {list(high_missing_cols)}")
            self.report.append(f"Dropped columns with >50% missing :{list(high_missing_cols)}")

        # Numeric columns
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                if self.df[col].skew() > 1:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                    print(f"Filled {col} with median.")
                    self.report.append(f"Filled {col} with median")
                else:
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                    print(f"Filled {col} with mean.")
                    self.report.append(f"Filled {col} with mean")

        # Categorical columns
        cat_cols = self.df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                print(f"📝 Filled {col} with mode.")
                self.report.append(f"Filled {col} with mode")

        # Missing indicators
        for col in self.df.columns:
            if self.df[col].isnull().mean() > 0.1:
                self.df[f"{col}_missing"] = self.df[col].isnull().astype(int)
                print(f"Added missing indicator for {col}.")
                self.report.append(f"Added missing indicator for {col}")

        # KNN for remaining numeric missing values
        if self.df.isnull().sum().sum() > 0 and len(numeric_cols) > 1:
            imputer = KNNImputer(n_neighbors=5)
            self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])
            print("🤝 Applied KNN imputation for remaining numeric missing values.")
            self.report.append(f"Applied KNN Imputer for remaining numeric missing values")

        # Now remove outliers
        self.remove_outliers()

    def remove_outliers(self):
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        before = len(self.df)

        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]

        after = len(self.df)
        print(f"Removed {before - after} outliers using IQR method.")
        self.report.append(f"Removed {before - after} outliers using IQR method")

    def prepare_for_ml(self):
        if self.df is None:
            print("No dataset loaded.")
            self.report.append("No dataset loaded")
            return

        # Separate features
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object']).columns.tolist()

        # Preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
            ]
        )

        # Apply transformation
        self.df = pd.DataFrame(
            preprocessor.fit_transform(self.df).toarray() if hasattr(preprocessor.fit_transform(self.df), "toarray") 
            else preprocessor.fit_transform(self.df)
        )
        print("Dataset transformed for Machine Learning (scaled + encoded).")
        self.report.append("Dataset transformed for Machine Learning (scaled + encoded)")

    def save_cleaned(self, suffix="cleaned"):
        if self.df is not None:
            cleaned_file = f"{suffix}_{self.file_name}"
            self.df.to_csv(cleaned_file, index=False)
            print(f"💾 Processed dataset saved as: {cleaned_file}")
            self.report.append(f"Processed dataset saved as: {cleaned_file}")
        else:
            print("⚠ No data to save.")
            self.report.append(f"No data to save")
    
    def get_report(self, as_text=True):
        if as_text:
            return "\n".join(self.report)   # text format (string)
        return self.report                 # list format


if __name__ == "__main__":
    cleaner = AutoDataCleaner()
    
    if cleaner.find_csv_file():
        cleaner.load_dataset()
        print("\nInitial Data Preview:")
        print(cleaner.preview())

        choice = input("\nDo you want to (1) Clean dataset or (2) Make dataset ready for ML model? Enter 1 or 2: ")

        if choice == "1":
            cleaner.analyse_missing_values()
            print("\nCleaned Data Preview:")
            print(cleaner.preview())
            cleaner.save_cleaned("cleaned")

        elif choice == "2":
            cleaner.analyse_missing_values()
            cleaner.prepare_for_ml()
            print("\nML-Ready Data Preview:")
            print(cleaner.preview())
            cleaner.save_cleaned("ml_ready")

        else:
            print("Invalid choice. Exiting.")
