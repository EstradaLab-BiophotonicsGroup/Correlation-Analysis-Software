import tkinter as tk
from first_file import run_pcf
from second_file import run_nandb

def main():
    root = tk.Tk()
    root.title("CASPY GUI")

    tk.Label(root, text="CASPY Analysis Suite", font=("Arial", 14)).pack(pady=10)

    # Buttons to trigger each analysis
    tk.Button(root, text="Run pCF", command=lambda: run_pcf(root)).pack(pady=10)
    tk.Button(root, text="Run NandB", command=lambda: run_nandb(root)).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
