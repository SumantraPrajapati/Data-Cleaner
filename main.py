# ui_app.py
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from cleaner import AutoDataCleaner   # import your main logic class
from tkinter import Toplevel, Text, Scrollbar, RIGHT, Y, BOTH, END

class DataApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto Data Cleaner")
        self.root.geometry("400x200")

        self.cleaner = AutoDataCleaner()

        self.label = tk.Label(root, text="Upload CSV File", font=("Arial", 14))
        self.label.pack(pady=10)

        self.upload_button = tk.Button(root, text="Choose File", command=self.load_csv)
        self.upload_button.pack(pady=10)

        self.file_label = tk.Label(root, text="No file selected", fg="gray")
        self.file_label.pack(pady=5)

    def load_csv(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv")]
        )

        if file_path:
            try:
                self.cleaner.file_name = file_path.split("/")[-1]
                self.cleaner.df = pd.read_csv(file_path)
                self.file_label.config(text=f"Loaded: {self.cleaner.file_name}", fg="green")
                self.open_preview_window()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV\n{e}")
        else:
            self.file_label.config(text="No file selected", fg="red")

    def open_preview_window(self):
        preview_win = tk.Toplevel(self.root)
        preview_win.title("Dataset Preview")

        # Show preview (first 5 rows)
        preview_text = tk.Text(preview_win, wrap="none", height=20, width=80)
        preview_text.insert(tk.END, self.cleaner.preview().to_string())
        preview_text.pack(padx=10, pady=10)

        label = tk.Label(preview_win, text="Choose an option:")
        label.pack(pady=5)

        btn_clean = tk.Button(preview_win, text="Clean Data", 
                              command=lambda: self.run_clean(preview_win))
        btn_clean.pack(side=tk.LEFT, padx=20, pady=10)

        btn_model = tk.Button(preview_win, text="Prepare for ML", 
                              command=lambda: self.run_ml(preview_win))
        btn_model.pack(side=tk.LEFT, padx=20, pady=10)

    def run_clean(self, win):
        self.cleaner.analyse_missing_values()
        win.destroy()
        self.ask_save_location("cleaned")
        show_report("\n".join(self.cleaner.report))

    def run_ml(self, win):
        self.cleaner.analyse_missing_values()
        self.cleaner.prepare_for_ml()
        win.destroy()
        self.ask_save_location("ml_ready")
        show_report("\n".join(self.cleaner.report)) 

    def ask_save_location(self, suffix):
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save Processed Dataset"
        )
        if save_path:
            self.cleaner.df.to_csv(save_path, index=False)
            messagebox.showinfo("Saved", f"Dataset saved at:\n{save_path}")
    

def show_report(report_text):
    report_window = Toplevel()
    report_window.title("Data Cleaning Report")

    text_widget = Text(report_window, wrap="word")
    text_widget.insert(END, report_text)
    text_widget.pack(expand=True, fill=BOTH)

    scrollbar = Scrollbar(report_window, command=text_widget.yview)
    scrollbar.pack(side=RIGHT, fill=Y)
    text_widget.config(yscrollcommand=scrollbar.set)


if __name__ == "__main__":
    root = tk.Tk()
    app = DataApp(root)
    root.mainloop()
