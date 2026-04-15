import tkinter as tk
from app import ModernCrayfishDetector


def main():
    root = tk.Tk()
    app = ModernCrayfishDetector(root)
    root.mainloop()


if __name__ == "__main__":
    main()