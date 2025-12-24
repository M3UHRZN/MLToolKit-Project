
# This file contains the Tkinter GUI (tabs, buttons, inputs, and plots).
# The GUI collects user choices (dataset, preprocessing, models, split ratio),
# then calls ml_core.py to train/evaluate and shows the results.
# ------------------------------

import threading  # Run model training in a background thread (keeps UI responsive)
import tkinter as tk  # Base Tkinter library
from tkinter import ttk, filedialog, messagebox  # ttk widgets + file picker + popups

import pandas as pd  # CSV reading + simple data inspection (dtypes, unique counts, etc.)

import matplotlib  # Plotting library (we embed plots into Tkinter)
matplotlib.use("TkAgg")  # Use TkAgg backend so Matplotlib can render inside Tkinter
from matplotlib.figure import Figure  # Matplotlib figure object (our confusion matrix plot)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Bridge: Matplotlib <-> Tkinter

from ui_helpers import ToolTip  # Small helper for hover tooltips
from ml_core import (
    TrainConfig, MLPParams,  # Dataclasses describing user choices / MLP settings
    recommend_target_column, dataset_summary, class_distribution_text,  # small utilities
    train_evaluate  # main function: build pipeline -> train -> compute metrics
)


class MLGuiApp(tk.Tk):
    """
    Main GUI application class.
    We inherit from tk.Tk (main window). Everything else is built inside this window.
    """

    def __init__(self):
        super().__init__()

        # Window properties
        self.title("GUI-Based ML Toolkit (Classification & Evaluation)")
        self.geometry("1300x850")
        self.minsize(980, 650)

        # Dataset state (loaded from CSV)
        self.df = None
        self.file_path = None

        # Confusion matrices are stored here after training:
        # model_name -> (confusion_matrix_array, label_list)
        self.cm_store = {}
        self._training_thread = None  # Background training thread handle

        # Build UI
        self._init_style()
        self._build_ui()

    # -------------------- UI style --------------------
    def _init_style(self):
        """
        Picks a decent ttk theme and sets some consistent spacing + fonts.
        (Pure UI improvement; no ML logic here.)
        """
        style = ttk.Style()

        # Try a few themes (available themes depend on OS)
        for theme in ("vista", "xpnative", "clam"):
            try:
                style.theme_use(theme)
                break
            except tk.TclError:
                continue

        # Small visual tweaks for a cleaner look
        style.configure("TLabel", padding=2)
        style.configure("TButton", padding=6)
        style.configure("TLabelframe", padding=8)
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"))

    # -------------------- Build UI --------------------
    def _build_ui(self):
        """Creates the overall layout: top header, tabs, and bottom action bar."""

        # Top header bar
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=(10, 6))

        ttk.Label(top, text="Classification & Evaluation Toolkit", style="Header.TLabel").pack(side="left")

        # Status text (we update this during load/train)
        self.status_var = tk.StringVar(value="Load a CSV to start.")
        ttk.Label(top, textvariable=self.status_var).pack(side="left", padx=18)

        # Help button (explains the flow)
        ttk.Button(top, text="How to use?", command=self.show_help).pack(side="right")

        # Notebook = tab system
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Tabs (Dataset / Settings / Results)
        self.tab_data = ttk.Frame(self.nb)
        self.tab_settings = ttk.Frame(self.nb)
        self.tab_results = ttk.Frame(self.nb)

        self.nb.add(self.tab_data, text="1) Dataset")
        self.nb.add(self.tab_settings, text="2) Settings")
        self.nb.add(self.tab_results, text="3) Results")

        # Build each tab separately
        self._build_tab_data()
        self._build_tab_settings()
        self._build_tab_results()

        # Bottom action bar: progress indicator + buttons
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=10, pady=(0, 10))

        # Progress bar (indeterminate = “busy spinner” style)
        self.progress = ttk.Progressbar(bottom, mode="indeterminate")
        self.progress.pack(side="left", fill="x", expand=True)

        # Main action buttons
        self.btn_train = ttk.Button(bottom, text="Train & Evaluate", command=self.train_and_evaluate_click)
        self.btn_train.pack(side="right", padx=(10, 0))

        self.btn_clear = ttk.Button(bottom, text="Clear Results", command=self.clear_results)
        self.btn_clear.pack(side="right")

        # Disable controls until a dataset is loaded
        self._set_controls_enabled(False)

    def _build_tab_data(self):
        """Dataset tab: load CSV, choose target, preview summary/distribution."""
        outer = ttk.Frame(self.tab_data)
        outer.pack(fill="both", expand=True, padx=10, pady=10)

        # Loader row (upload button + target dropdown)
        row = ttk.LabelFrame(outer, text="CSV Loader")
        row.pack(fill="x")

        self.btn_upload = ttk.Button(row, text="Upload CSV", command=self.load_csv)
        self.btn_upload.pack(side="left", padx=6, pady=6)
        ToolTip(self.btn_upload, "Select a CSV file. First row should contain column names.")

        # Show file name after loading
        self.lbl_file = ttk.Label(row, text="No file selected")
        self.lbl_file.pack(side="left", padx=10)

        # Target column selector (label column)
        ttk.Label(row, text="Target (label) column:").pack(side="left", padx=(25, 6))
        self.target_var = tk.StringVar(value="")
        self.target_combo = ttk.Combobox(row, textvariable=self.target_var, state="readonly", width=26)
        self.target_combo.pack(side="left")
        ToolTip(self.target_combo, "Choose the label/target column to predict (e.g., risk, class).")

        # “Auto-pick” uses a heuristic to choose a likely label column
        self.btn_auto_target = ttk.Button(row, text="Auto-pick", command=self.auto_pick_target)
        self.btn_auto_target.pack(side="left", padx=6)

        # Preview a few rows in the log area (useful for debugging)
        self.btn_preview = ttk.Button(row, text="Preview head()", command=self.preview_head)
        self.btn_preview.pack(side="right", padx=6)

        # Dataset summary section (left: summary, right: class distribution)
        info = ttk.LabelFrame(outer, text="Dataset Summary")
        info.pack(fill="both", expand=True, pady=(10, 0))

        left = ttk.Frame(info)
        left.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        # Text widget: dataset info (shape, missing, numeric/cat columns)
        self.summary_text = tk.Text(left, height=14, wrap="word")
        self.summary_text.pack(fill="both", expand=True)
        self._set_text(self.summary_text, "Load a dataset to see summary here.\n", readonly=True)

        right = ttk.Frame(info)
        right.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        ttk.Label(right, text="Class distribution (after selecting target):").pack(anchor="w", pady=(0, 6))

        # Text widget: class distribution of the selected target
        self.dist_text = tk.Text(right, height=14, wrap="word")
        self.dist_text.pack(fill="both", expand=True)
        self._set_text(self.dist_text, "Select target to see class distribution.\n", readonly=True)

        # When user changes target, update distribution + binning UI state
        self.target_combo.bind(
            "<<ComboboxSelected>>",
            lambda e: (self.update_class_distribution(), self._sync_binning_state())
        )

    def _build_tab_settings(self):
        """Settings tab: preprocessing, split ratio, model selection, MLP + binning options."""
        outer = ttk.Frame(self.tab_settings)
        outer.pack(fill="both", expand=True, padx=10, pady=10)

        # Configure columns so panels resize nicely
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)
        outer.columnconfigure(2, weight=1)

        # ---------------- Preprocessing panel ----------------
        pre = ttk.LabelFrame(outer, text="Preprocessing")
        pre.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))

        self.use_onehot = tk.BooleanVar(value=True)  # encode categorical features
        self.use_norm = tk.BooleanVar(value=True)    # scale numeric features

        chk_oh = ttk.Checkbutton(pre, text="One-Hot Encoding (categorical features)", variable=self.use_onehot)
        chk_oh.pack(anchor="w", padx=8, pady=(8, 4))
        ToolTip( chk_oh,"Enable this if your input FEATURES (X) contain categorical/string values (excluding the target).")

        chk_norm = ttk.Checkbutton(pre, text="Normalization (numeric)", variable=self.use_norm, command=self._sync_scaler_state)
        chk_norm.pack(anchor="w", padx=8, pady=4)
        ToolTip(chk_norm, "Scale numeric features. Helpful for Perceptron/MLP.")

        ttk.Label(pre, text="Scaler:").pack(anchor="w", padx=8, pady=(10, 2))
        self.scaler_type = tk.StringVar(value="standard")
        self.scaler_combo = ttk.Combobox(
            pre, state="readonly",
            values=["standard", "minmax"],
            textvariable=self.scaler_type,
            width=12
        )
        self.scaler_combo.pack(anchor="w", padx=8, pady=(0, 8))
        ToolTip(self.scaler_combo, "standard=StandardScaler, minmax=MinMaxScaler")

        # Enable/disable scaler dropdown depending on “Normalization” checkbox
        self._sync_scaler_state()

        # ---------------- Train/Test split panel ----------------
        split = ttk.LabelFrame(outer, text="Train/Test Split")
        split.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=(0, 8))

        ttk.Label(split, text="Test size (0.10 to 0.50):").pack(anchor="w", padx=8, pady=(8, 2))
        self.test_size_var = tk.DoubleVar(value=0.2)
        self.test_slider = ttk.Scale(split, from_=0.1, to=0.5, variable=self.test_size_var, command=self._update_split_label)
        self.test_slider.pack(fill="x", padx=8, pady=6)
        self.lbl_split = ttk.Label(split, text="Current test size: 0.20 (Train 0.80)")
        self.lbl_split.pack(anchor="w", padx=8, pady=(0, 8))

        # Random state makes train/test split reproducible
        rs_row = ttk.Frame(split)
        rs_row.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Label(rs_row, text="Random state:").pack(side="left")
        self.random_state_var = tk.IntVar(value=42)
        ttk.Entry(rs_row, textvariable=self.random_state_var, width=10).pack(side="left", padx=6)

        # ---------------- Target binning panel ----------------
        # Binning converts numeric target (age/income/etc.) into a few categories (3/5/7 classes)
        target_opt = ttk.LabelFrame(outer, text="Target Options (for numeric targets)")
        target_opt.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))

        self.bin_target_var = tk.BooleanVar(value=False)  # whether to bin numeric target
        self.bin_bins_var = tk.IntVar(value=3)            # number of bins/classes
        self.bin_strategy_var = tk.StringVar(value="quantile")  # quantile/uniform

        chk_bin = ttk.Checkbutton(
            target_opt,
            text="Discretize numeric target (binning)",
            variable=self.bin_target_var,
            command=self._sync_binning_state
        )
        chk_bin.pack(anchor="w", padx=8, pady=(8, 4))
        ToolTip(chk_bin, "If the selected target is numeric (e.g., age), convert it into a few classes (bins).")

        rowb = ttk.Frame(target_opt)
        rowb.pack(fill="x", padx=8, pady=(4, 8))

        ttk.Label(rowb, text="Bins:").pack(side="left")
        self.bin_bins_combo = ttk.Combobox(
            rowb, state="readonly", width=6, values=[3, 5, 7], textvariable=self.bin_bins_var
        )
        self.bin_bins_combo.pack(side="left", padx=6)

        ttk.Label(rowb, text="Strategy:").pack(side="left", padx=(18, 0))
        self.bin_strategy_combo = ttk.Combobox(
            rowb, state="readonly", width=10, values=["quantile", "uniform"], textvariable=self.bin_strategy_var
        )
        self.bin_strategy_combo.pack(side="left", padx=6)

        ToolTip(self.bin_bins_combo, "Number of classes after discretization (3/5/7).")
        ToolTip(self.bin_strategy_combo, "quantile: equal-frequency bins, uniform: equal-width bins.")

        # ---------------- Models panel ----------------
        models = ttk.LabelFrame(outer, text="Models")
        models.grid(row=0, column=2, sticky="nsew", pady=(0, 8))

        self.use_perc = tk.BooleanVar(value=True)
        self.use_mlp = tk.BooleanVar(value=True)
        self.use_tree = tk.BooleanVar(value=True)

        chk_p = ttk.Checkbutton(models, text="Perceptron", variable=self.use_perc)
        chk_p.pack(anchor="w", padx=8, pady=(8, 2))
        chk_m = ttk.Checkbutton(models, text="MLP (Backprop)", variable=self.use_mlp, command=self._toggle_mlp_panel)
        chk_m.pack(anchor="w", padx=8, pady=2)
        chk_t = ttk.Checkbutton(models, text="Decision Tree", variable=self.use_tree)
        chk_t.pack(anchor="w", padx=8, pady=2)

        ToolTip(chk_p, "Fast linear classifier. Works better with normalization.")
        ToolTip(chk_m, "Neural network. Needs scaling; can model nonlinear patterns.")
        ToolTip(chk_t, "Tree-based model. Scaling not mandatory, but OK to keep pipeline consistent.")

        # MLP hyperparameter panel (hidden layer sizes, activation, etc.)
        # Placed at row=2 to avoid overlapping with target_opt (row=1).
        self.mlp_frame = ttk.LabelFrame(outer, text="MLP Settings (Hidden layers = 1,2,3 or 4)")
        self.mlp_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        self._build_mlp_panel(self.mlp_frame)

        # Initial enable/disable state for binning controls
        self._sync_binning_state()

    def _build_mlp_panel(self, parent):
        """Builds the MLP settings area (layer count + neuron counts + optimizer settings)."""
        row1 = ttk.Frame(parent)
        row1.pack(fill="x", padx=8, pady=(8, 6))

        ttk.Label(row1, text="Hidden layer count:").pack(side="left")
        self.mlp_layers_var = tk.IntVar(value=2)
        self.mlp_layers_combo = ttk.Combobox(row1, state="readonly", width=5, values=[1, 2, 3, 4], textvariable=self.mlp_layers_var)
        self.mlp_layers_combo.pack(side="left", padx=6)
        self.mlp_layers_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_mlp_neurons_inputs())

        # Neurons per layer (L3 is disabled if hidden_layers == 2)
        ttk.Label(row1, text="Neurons L1:").pack(side="left", padx=(20, 4))
        self.mlp_n1 = tk.IntVar(value=32)
        ttk.Entry(row1, textvariable=self.mlp_n1, width=8).pack(side="left", padx=4)

        ttk.Label(row1, text="Neurons L2:").pack(side="left", padx=(20, 4))
        self.mlp_n2 = tk.IntVar(value=16)
        self.ent_n2 = ttk.Entry(row1, textvariable=self.mlp_n2, width=8)
        self.ent_n2.pack(side="left", padx=4)

        ttk.Label(row1, text="Neurons L3:").pack(side="left", padx=(20, 4))
        self.mlp_n3 = tk.IntVar(value=8)
        self.ent_n3 = ttk.Entry(row1, textvariable=self.mlp_n3, width=8)
        self.ent_n3.pack(side="left", padx=4)

        ttk.Label(row1, text="Neurons L4:").pack(side="left", padx=(20, 4))
        self.mlp_n4 = tk.IntVar(value=4)
        self.ent_n4 = ttk.Entry(row1, textvariable=self.mlp_n4, width=8)
        self.ent_n4.pack(side="left", padx=4)

        row2 = ttk.Frame(parent)
        row2.pack(fill="x", padx=8, pady=(0, 10))

        # Activation function for the neural network
        ttk.Label(row2, text="Activation:").pack(side="left")
        self.mlp_activation = tk.StringVar(value="relu")
        ttk.Combobox(row2, state="readonly", width=10, values=["relu", "tanh", "logistic"], textvariable=self.mlp_activation).pack(
            side="left", padx=6
        )

        # Learning rate + max iterations are standard MLPClassifier hyperparameters
        ttk.Label(row2, text="Learning rate init:").pack(side="left", padx=(20, 0))
        self.mlp_lr = tk.DoubleVar(value=0.001)
        ttk.Entry(row2, textvariable=self.mlp_lr, width=10).pack(side="left", padx=6)

        ttk.Label(row2, text="Max iter:").pack(side="left", padx=(20, 0))
        self.mlp_max_iter = tk.IntVar(value=500)
        ttk.Entry(row2, textvariable=self.mlp_max_iter, width=10).pack(side="left", padx=6)

        # Apply initial enabled/disabled state for L3
        self._refresh_mlp_neurons_inputs()

    def _build_tab_results(self):
        """Results tab: metrics table + confusion matrix plot + run log."""
        outer = ttk.Frame(self.tab_results)
        outer.pack(fill="both", expand=True, padx=10, pady=10)

        outer.rowconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)

        # Metrics table (Treeview)
        table_frame = ttk.LabelFrame(outer, text="Metrics (weighted average)")
        table_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))

        cols = ("Model", "Accuracy", "Precision", "Recall", "F1-Score")
        self.metrics_tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=9)
        for c in cols:
            self.metrics_tree.heading(c, text=c)
            self.metrics_tree.column(c, anchor="center", width=130 if c != "Model" else 160)

        yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.metrics_tree.yview)
        self.metrics_tree.configure(yscrollcommand=yscroll.set)

        self.metrics_tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        yscroll.pack(side="right", fill="y", pady=8, padx=(0, 8))

        # Confusion matrix area (Matplotlib canvas)
        cm_frame = ttk.LabelFrame(outer, text="Confusion Matrix")
        cm_frame.grid(row=0, column=1, sticky="nsew", pady=(0, 8))

        cm_top = ttk.Frame(cm_frame)
        cm_top.pack(fill="x", padx=8, pady=(8, 6))

        ttk.Label(cm_top, text="Select model:").pack(side="left")
        self.cm_model_var = tk.StringVar(value="")
        self.cm_combo = ttk.Combobox(cm_top, state="readonly", width=24, textvariable=self.cm_model_var, values=[])
        self.cm_combo.pack(side="left", padx=6)

        ttk.Button(cm_top, text="Render", command=self.render_confusion_matrix).pack(side="left", padx=6)

        # Matplotlib figure embedded into Tkinter
        self.fig = Figure(figsize=(4.6, 3.4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("No confusion matrix yet")
        self.canvas = FigureCanvasTkAgg(self.fig, master=cm_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Log area (human-readable training notes)
        log_frame = ttk.LabelFrame(outer, text="Run Log")
        log_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")

        self.txt_log = tk.Text(log_frame, height=10, wrap="word")
        self.txt_log.pack(fill="both", expand=True, padx=8, pady=8)
        self.log("Load dataset → choose target → set options → Train & Evaluate")

    # -------------------- small UI helpers --------------------
    def _set_text(self, widget: tk.Text, text: str, readonly: bool = True):
        """Write text into a Tkinter Text widget. If readonly=True, disable editing."""
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        if readonly:
            widget.configure(state="disabled")

    def log(self, msg: str):
        """Append a line to the log text box (Results tab)."""
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")

    def set_status(self, msg: str):
        """Update the top status label."""
        self.status_var.set(msg)

    def _set_controls_enabled(self, enabled: bool):
        """Enable/disable controls that require a loaded dataset."""
        state = "normal" if enabled else "disabled"
        self.btn_train.configure(state=state)
        self.btn_clear.configure(state=state)
        self.target_combo.configure(state="readonly" if enabled else "disabled")
        self.btn_auto_target.configure(state=state)
        self.btn_preview.configure(state=state)

    def _sync_scaler_state(self):
        """If normalization is disabled, scaler dropdown should be disabled too."""
        self.scaler_combo.configure(state="readonly" if self.use_norm.get() else "disabled")

    def _update_split_label(self, _=None):
        """Update the label under the slider to show current train/test split."""
        t = float(self.test_size_var.get())
        self.lbl_split.config(text=f"Current test size: {t:.2f} (Train {1-t:.2f})")

    def _toggle_mlp_panel(self):
        """Show/hide the MLP settings panel when the MLP checkbox changes."""
        if self.use_mlp.get():
            self.mlp_frame.grid()
        else:
            self.mlp_frame.grid_remove()

    def _refresh_mlp_neurons_inputs(self):
        layers = int(self.mlp_layers_var.get())

         # L1 always enabled (we don't disable it)
        self.ent_n2.configure(state="normal" if layers >= 2 else "disabled")
        self.ent_n3.configure(state="normal" if layers >= 3 else "disabled")
        self.ent_n4.configure(state="normal" if layers >= 4 else "disabled")

    def _sync_binning_state(self):
        """
        Enable/disable binning controls.
        - If no dataset: keep disabled.
        - If target is not numeric: force binning off.
        - If target numeric: enable controls only when checkbox is ON.
        """
        if self.df is None:
            try:
                self.bin_bins_combo.configure(state="disabled")
                self.bin_strategy_combo.configure(state="disabled")
            except Exception:
                pass
            return

        target = self.target_var.get().strip()
        is_numeric = False
        if target in self.df.columns:
            is_numeric = pd.api.types.is_numeric_dtype(self.df[target])

        if not is_numeric:
            # Non-numeric labels (e.g., risk/class) do not need binning
            self.bin_target_var.set(False)
            self.bin_bins_combo.configure(state="disabled")
            self.bin_strategy_combo.configure(state="disabled")
            return

        # Numeric target: allow binning options when checkbox is enabled
        if self.bin_target_var.get():
            self.bin_bins_combo.configure(state="readonly")
            self.bin_strategy_combo.configure(state="readonly")
        else:
            self.bin_bins_combo.configure(state="disabled")
            self.bin_strategy_combo.configure(state="disabled")

    # -------------------- Dataset actions --------------------
    def load_csv(self):
        """Open file dialog -> read CSV -> fill UI controls -> show summary."""
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read CSV:\n{e}")
            return

        # Basic sanity check: we need at least 1 feature + 1 target column
        if df.shape[1] < 2:
            messagebox.showerror("Error", "Dataset must have at least 2 columns (features + target).")
            return

        self.df = df
        self.file_path = path
        self.lbl_file.config(text=path.split("/")[-1])

        # Populate target dropdown
        cols = list(df.columns)
        self.target_combo["values"] = cols

        # Choose a recommended target based on simple heuristics
        rec = recommend_target_column(df)
        self.target_var.set(rec)

        # Update binning controls based on target dtype
        self._sync_binning_state()

        # Enable UI controls now that we have data
        self._set_controls_enabled(True)

        self.set_status(f"Loaded: {path.split('/')[-1]}  |  Rows={df.shape[0]}, Cols={df.shape[1]}")

        # Display dataset summary + class distribution
        self._set_text(self.summary_text, dataset_summary(df, path), readonly=True)
        self.update_class_distribution()

        # Log important info (nice for report screenshots too)
        self.log("======================================")
        self.log(f"Loaded dataset: {path}")
        self.log(f"Shape: {df.shape}")
        self.log(f"Recommended target: {rec}")

        # Move user to Settings tab (next step)
        self.nb.select(self.tab_settings)

    def auto_pick_target(self):
        """Pick a target automatically and refresh dependent UI."""
        if self.df is None:
            return
        rec = recommend_target_column(self.df)
        self.target_var.set(rec)
        self.update_class_distribution()
        self._sync_binning_state()  # target dtype might change
        self.log(f"Auto-picked target: {rec}")

    def preview_head(self):
        """Print first rows into the log (quick check if parsing is correct)."""
        if self.df is None:
            return
        self.log("---- head(10) ----")
        self.log(str(self.df.head(10)))

    def update_class_distribution(self):
        """Update right-side text area showing class counts for selected target."""
        if self.df is None:
            return
        target = self.target_var.get().strip()
        if not target or target not in self.df.columns:
            return
        self._set_text(self.dist_text, class_distribution_text(self.df, target), readonly=True)

    # -------------------- Training (threaded) --------------------
    def train_and_evaluate_click(self):
        """
        Validate user inputs and start training in a background thread.
        Background thread is used so the GUI never freezes during training.
        """
        if self.df is None:
            messagebox.showwarning("Warning", "Please upload a CSV dataset first.")
            return

        # Don’t start a second run while one is still running
        if self._training_thread and self._training_thread.is_alive():
            messagebox.showinfo("Info", "Training is already running.")
            return

        target_col = self.target_var.get().strip()
        if not target_col or target_col not in self.df.columns:
            messagebox.showwarning("Warning", "Please select a valid target column.")
            return

        # Numeric target safeguard: encourage binning for classification (avoid huge class counts)
        if pd.api.types.is_numeric_dtype(self.df[target_col]):
            uniq = self.df[target_col].dropna().nunique()
            if uniq > 25 and not self.bin_target_var.get():
                messagebox.showwarning(
                    "Numeric target detected",
                    f"'{target_col}' has {uniq} unique values.\n"
                    "For classification, numeric targets should be discretized (binned).\n\n"
                    "Please enable: Discretize numeric target (binning)."
                )
                return

        # UI feedback: show “busy” state
        self.progress.start(10)
        self.btn_train.configure(state="disabled")
        self.set_status("Training... please wait")

        # Run training in another thread
        self._training_thread = threading.Thread(target=self._train_worker, daemon=True)
        self._training_thread.start()

    def _train_worker(self):
        """
        Runs model training + evaluation.
        IMPORTANT: This runs in a background thread. UI updates must be scheduled via self.after().
        """
        try:
            # Package all UI settings into a single configuration object
            cfg = TrainConfig(
                target_col=self.target_var.get().strip(),
                test_size=float(self.test_size_var.get()),
                random_state=int(self.random_state_var.get()),
                use_onehot=bool(self.use_onehot.get()),
                use_norm=bool(self.use_norm.get()),
                scaler_type=str(self.scaler_type.get()),
                use_perceptron=bool(self.use_perc.get()),
                use_mlp=bool(self.use_mlp.get()),
                use_tree=bool(self.use_tree.get()),

                # Binning options (only relevant for numeric targets)
                bin_numeric_target=bool(self.bin_target_var.get()),
                bin_count=int(self.bin_bins_var.get()),
                bin_strategy=str(self.bin_strategy_var.get()),

                # MLP hyperparameters
                mlp=MLPParams(
                    hidden_layers=int(self.mlp_layers_var.get()),
                    n1=int(self.mlp_n1.get()),
                    n2=int(self.mlp_n2.get()),
                    n3=int(self.mlp_n3.get()),
                    n4=int(self.mlp_n4.get()),
                    activation=str(self.mlp_activation.get()),
                    learning_rate_init=float(self.mlp_lr.get()),
                    max_iter=int(self.mlp_max_iter.get())
                )

            )

            # Core ML call: returns metrics + confusion matrices + meta info
            results, cm_store, meta = train_evaluate(self.df, cfg)

            # Schedule UI update safely on main Tkinter thread
            self.after(0, lambda: self._update_results_ui(results, cm_store, cfg, meta))

        except Exception as e:
            # Any error during training should be shown to the user as a popup
            err = str(e)
            self.after(0, self._on_train_error, err)

    def _update_results_ui(self, results, cm_store, cfg: TrainConfig, meta: dict):
        """Fill the results table, store confusion matrices, update status/log, render initial CM."""
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)

        # Save confusion matrices for plotting later
        self.cm_store = cm_store

        # Populate dropdown with available models
        self.cm_combo["values"] = list(self.cm_store.keys())
        if self.cm_store:
            self.cm_model_var.set(list(self.cm_store.keys())[0])

        # Determine "best" model by F1 score (weighted)
        best_name = max(results, key=lambda x: x[4])[0] if results else None
        self.metrics_tree.tag_configure("best", background="#e8f4ff")

        # Insert rows into the table
        for (name, acc, prec, rec, f1) in results:
            tag = "best" if name == best_name else ""
            self.metrics_tree.insert(
                "", "end",
                values=(name, f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"),
                tags=(tag,)
            )

        # Log run configuration (useful for report)
        self.log("======================================")
        self.log(f"Target: {cfg.target_col} (bin_numeric_target={cfg.bin_numeric_target}, bins={cfg.bin_count}, strategy={cfg.bin_strategy})")
        self.log(f"Numeric cols: {len(meta['num_cols'])} | Categorical cols: {len(meta['cat_cols'])}")
        self.log(f"OneHot={cfg.use_onehot} | Norm={cfg.use_norm} ({cfg.scaler_type})")
        self.log(f"Split: Train={1-cfg.test_size:.2f} Test={cfg.test_size:.2f} | RandomState={cfg.random_state}")
        for (name, acc, prec, rec, f1) in results:
            star = " ⭐" if name == best_name else ""
            self.log(f"[{name}]{star} acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

        # Reset UI busy state
        self.set_status(f"Done. Best by F1: {best_name}")
        self.progress.stop()
        self.btn_train.configure(state="normal")

        # Jump to Results tab and show the first confusion matrix
        self.nb.select(self.tab_results)
        self.render_confusion_matrix()

    def _on_train_error(self, msg: str):
        """Stop progress bar and show training error message."""
        self.progress.stop()
        self.btn_train.configure(state="normal")
        self.set_status("Error")
        messagebox.showerror("Error", msg)

    def clear_results(self):
        """Clear table, plots, and log."""
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)

        self.cm_store.clear()
        self.cm_combo["values"] = []
        self.cm_model_var.set("")

        self.ax.clear()
        self.ax.set_title("No confusion matrix yet")
        self.canvas.draw()

        self.txt_log.delete("1.0", "end")
        self.log("Results cleared.")
        self.set_status("Results cleared.")

    # -------------------- Confusion Matrix render --------------------
    def render_confusion_matrix(self):
        """
        Draw selected model’s confusion matrix as a heatmap.
        Note: too many classes (e.g., raw 'age') makes the plot unreadable,
        so we show a friendly message in that case.
        """
        name = self.cm_model_var.get().strip()
        if not name or name not in self.cm_store:
            return

        cm, labels = self.cm_store[name]

        # If class count is huge, show a warning text instead of a messy plot
        if len(labels) > 25:
            self.ax.clear()
            self.ax.axis("off")
            self.ax.text(
                0.5, 0.5,
                f"Too many classes ({len(labels)}) to display.\n"
                "Enable binning for numeric targets or choose a categorical target.",
                ha="center", va="center", transform=self.ax.transAxes
            )
            self.canvas.draw()
            return

        # Simple heatmap view (numbers printed on cells)
        self.ax.clear()
        self.ax.imshow(cm)
        self.ax.set_title(f"Confusion Matrix - {name}")
        self.ax.set_xlabel("Predicted")
        self.ax.set_ylabel("Actual")

        # Tick labels show class names
        self.ax.set_xticks(range(len(labels)))
        self.ax.set_yticks(range(len(labels)))
        self.ax.set_xticklabels(labels, rotation=45, ha="right")
        self.ax.set_yticklabels(labels)

        # Write counts into each cell
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                self.ax.text(j, i, str(cm[i, j]), ha="center", va="center")

        self.fig.tight_layout()
        self.canvas.draw()

    # -------------------- Help --------------------
    def show_help(self):
        """Short usage guide for first-time users."""
        msg = (
            "1) Dataset tab → Upload CSV\n"
            "2) Choose Target (label) column (e.g., 'risk')\n"
            "3) Settings tab → Select preprocessing + models + split ratio\n"
            "4) Train & Evaluate\n"
            "5) Results tab → metrics table + confusion matrix\n\n"
            "Tip: If your target is numeric (e.g., age), enable binning to convert it into classes."
        )
        messagebox.showinfo("How to use", msg)


if __name__ == "__main__":
    # Entry point: start the GUI application loop
    app = MLGuiApp()
    app.mainloop()
