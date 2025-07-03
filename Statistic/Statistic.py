import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

class StatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Paper Stats with Python")
        self.df = None

        # Data input mode selection
        frame_mode = ttk.Frame(root, padding=10)
        frame_mode.pack(fill='x')
        ttk.Label(frame_mode, text="Data Input Mode:").pack(side='left')
        self.mode_var = tk.StringVar(value="Excel")
        ttk.Radiobutton(frame_mode, text="Excel", variable=self.mode_var, value="Excel", command=self.update_data_mode).pack(side='left')
        ttk.Radiobutton(frame_mode, text="Manual", variable=self.mode_var, value="Manual", command=self.update_data_mode).pack(side='left')

        # File upload frame
        self.frame_file = ttk.Frame(root, padding=10)
        self.frame_file.pack(fill='x')
        ttk.Button(self.frame_file, text="Upload Excel", command=self.load_file).pack(side='left')
        self.file_label = ttk.Label(self.frame_file, text="No file loaded")
        self.file_label.pack(side='left', padx=10)

        # Manual entry frame
        self.frame_manual = ttk.Frame(root, padding=10)
        ttk.Label(self.frame_manual, text="Number of Entries:").grid(row=0, column=0)
        self.num_entry = ttk.Entry(self.frame_manual, width=5)
        self.num_entry.grid(row=0, column=1, padx=5)
        ttk.Label(self.frame_manual, text="Numeric Column Name:").grid(row=0, column=2)
        self.num_name = ttk.Entry(self.frame_manual)
        self.num_name.grid(row=0, column=3, padx=5)
        ttk.Label(self.frame_manual, text="Group Column Name:").grid(row=0, column=4)
        self.grp_name = ttk.Entry(self.frame_manual)
        self.grp_name.grid(row=0, column=5, padx=5)
        ttk.Button(self.frame_manual, text="Create Manual Input", command=self.create_manual_inputs).grid(row=0, column=6, padx=5)

        # Analysis selection
        frame_anal = ttk.Frame(root, padding=10)
        frame_anal.pack(fill='x')
        ttk.Label(frame_anal, text="Analysis Type:").pack(side='left')
        self.analysis_var = tk.StringVar()
        self.analysis_box = ttk.OptionMenu(frame_anal, self.analysis_var, "",
                                           "Descriptive", "Group Comparison", "Boxplot", "Histogram", "Regression", command=self.update_params)
        self.analysis_box.pack(side='left', padx=5)

        # Parameters frame
        self.param_frame = ttk.Frame(root, padding=10)
        self.param_frame.pack(fill='x')

        # Run button
        ttk.Button(root, text="Run Analysis", command=self.run_analysis).pack(pady=5)

        # Text output
        self.text = tk.Text(root, height=10)
        self.text.pack(fill='both', expand=True, padx=10, pady=5)

        # Initialize mode
        self.update_data_mode()

    def update_data_mode(self):
        mode = self.mode_var.get()
        if mode == "Excel":
            self.frame_file.pack(fill='x')
            self.frame_manual.forget()
        else:
            self.frame_file.forget()
            self.frame_manual.pack(fill='x')

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls")])
        if not path:
            return
        try:
            self.df = pd.read_excel(path)
            self.file_label.config(text=path.split('/')[-1])
            messagebox.showinfo("Loaded", f"Data loaded with {len(self.df)} rows and {len(self.df.columns)} columns.")
            self.update_params()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def create_manual_inputs(self):
        try:
            n = int(self.num_entry.get())
            num_col = self.num_name.get().strip()
            grp_col = self.grp_name.get().strip()
            if n <= 0 or not num_col or not grp_col:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Please enter valid number and column names.")
            return
        # Build manual input window
        win = tk.Toplevel(self.root)
        win.title("Manual Data Entry")
        entries = []
        ttk.Label(win, text=grp_col).grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(win, text=num_col).grid(row=0, column=1, padx=5, pady=5)
        for i in range(n):
            e1 = ttk.Entry(win, width=15)
            e1.grid(row=i+1, column=0, padx=5, pady=2)
            e2 = ttk.Entry(win, width=15)
            e2.grid(row=i+1, column=1, padx=5, pady=2)
            entries.append((e1, e2))
        def load_manual():
            data = {grp_col: [], num_col: []}
            try:
                for g_e, n_e in entries:
                    data[grp_col].append(g_e.get())
                    data[num_col].append(float(n_e.get()))
                self.df = pd.DataFrame(data)
                self.file_label.config(text="Manual Data")
                messagebox.showinfo("Loaded", f"Manual data loaded with {len(self.df)} rows.")
                self.update_params()
                win.destroy()
            except Exception:
                messagebox.showerror("Error", "Please fill all entries correctly.")
        ttk.Button(win, text="Load Manual", command=load_manual).grid(row=n+1, column=0, columnspan=2, pady=10)

    def update_params(self, *args):
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        if self.df is None:
            return
        numeric = list(self.df.select_dtypes(include=[np.number]).columns)
        categorical = list(self.df.select_dtypes(exclude=[np.number]).columns)
        choice = self.analysis_var.get()
        if choice in ("Group Comparison", "Boxplot"):
            ttk.Label(self.param_frame, text="Numeric Column:").grid(row=0, column=0)
            self.num_var = tk.StringVar()
            ttk.OptionMenu(self.param_frame, self.num_var, numeric[0] if numeric else '', *numeric).grid(row=0, column=1)
            ttk.Label(self.param_frame, text="Group Column:").grid(row=1, column=0)
            self.grp_var = tk.StringVar()
            ttk.OptionMenu(self.param_frame, self.grp_var, categorical[0] if categorical else '', *categorical).grid(row=1, column=1)
        elif choice == "Histogram":
            ttk.Label(self.param_frame, text="Numeric Column:").grid(row=0, column=0)
            self.num_var = tk.StringVar()
            ttk.OptionMenu(self.param_frame, self.num_var, numeric[0] if numeric else '', *numeric).grid(row=0, column=1)
            ttk.Label(self.param_frame, text="Bins:").grid(row=1, column=0)
            self.bins_entry = ttk.Entry(self.param_frame)
            self.bins_entry.insert(0, '10')
            self.bins_entry.grid(row=1, column=1)
        elif choice == "Regression":
            ttk.Label(self.param_frame, text="Dependent (Numeric):").grid(row=0, column=0)
            self.dep_var = tk.StringVar()
            ttk.OptionMenu(self.param_frame, self.dep_var, numeric[0] if numeric else '', *numeric).grid(row=0, column=1)
            ttk.Label(self.param_frame, text="Independents:").grid(row=1, column=0)
            self.listbox = tk.Listbox(self.param_frame, selectmode='multiple', height=6)
            for col in self.df.columns:
                if col != self.dep_var.get():
                    self.listbox.insert('end', col)
            self.listbox.grid(row=1, column=1)

    def run_analysis(self):
        if self.df is None:
            messagebox.showwarning("No data", "Please load or enter data first.")
            return
        choice = self.analysis_var.get()
        self.text.delete(1.0, 'end')
        if choice == "Descriptive":
            desc = self.df.describe(include='all').round(3)
            self.text.insert('end', desc.to_string())
        elif choice == "Group Comparison":
            col = self.num_var.get(); grp = self.grp_var.get()
            groups = self.df.groupby(grp)[col]
            names = list(groups.groups.keys())
            if len(names) != 2:
                messagebox.showerror("Error", "Require exactly 2 groups.")
                return
            a, b = groups.get_group(names[0]), groups.get_group(names[1])
            p1 = stats.shapiro(a).pvalue; p2 = stats.shapiro(b).pvalue
            if p1 > .05 and p2 > .05:
                stat, p = stats.ttest_ind(a, b); test = 't-test'
            else:
                stat, p = stats.mannwhitneyu(a, b); test = 'Mann-Whitney U'
            self.text.insert('end', f"Test: {test}\nStatistic: {stat:.3f}, p-value: {p:.4f}\n")
        elif choice == "Boxplot":
            col = self.num_var.get(); grp = self.grp_var.get()
            plt.figure()
            self.df.boxplot(column=col, by=grp)
            plt.title(f"Boxplot of {col} by {grp}")
            plt.suptitle('')
            plt.show()
        elif choice == "Histogram":
            col = self.num_var.get(); bins = int(self.bins_entry.get())
            plt.figure()
            self.df[col].hist(bins=bins)
            plt.title(f"Histogram of {col}")
            plt.xlabel(col); plt.ylabel('Count')
            plt.show()
        elif choice == "Regression":
            dep = self.dep_var.get()
            sel = [self.listbox.get(i) for i in self.listbox.curselection()]
            if not sel:
                messagebox.showerror("Error", "Select at least one independent.")
                return
            formula = f"{dep} ~ {' + '.join(sel)}"
            model = smf.ols(formula, data=self.df).fit()
            self.text.insert('end', model.summary().as_text())
        else:
            messagebox.showwarning("Select", "Select an analysis type.")

if __name__ == '__main__':
    root = tk.Tk()
    app = StatApp(root)
    root.mainloop()
