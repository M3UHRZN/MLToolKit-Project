# ------------------------------
# ui_helpers.py
# ------------------------------
# Small UI helpers that are used by the GUI (app.py).
# For now, it contains a minimal tooltip implementation.
# ------------------------------

import tkinter as tk  # Tkinter primitives (Toplevel window, Label, etc.)


class ToolTip:
    """
    Very small tooltip helper for Tkinter widgets.
    When the mouse enters a widget, it shows a small popup.
    When the mouse leaves, it hides the popup.
    """
    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tip = None  # Holds the current tooltip window (if shown)

        # Bind mouse hover events
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _=None):
        """Create and display the tooltip window near the widget."""
        if self.tip or not self.text:
            return

        # Position tooltip slightly below/right of the widget
        x = self.widget.winfo_rootx() + 15
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 10

        # Toplevel = small floating window
        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)  # Remove window border
        tw.wm_geometry(f"+{x}+{y}")

        # Label showing the tooltip text
        lbl = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#1f1f1f",
            foreground="white",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=6
        )
        lbl.pack()

    def hide(self, _=None):
        """Destroy tooltip window if it exists."""
        if self.tip:
            self.tip.destroy()
            self.tip = None
