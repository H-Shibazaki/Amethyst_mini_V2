"""GUIの全て。リアルタイム推論専用 (mini版)
   学習／ファインチューニング機能はコメントアウトしています。
"""
import tkinter as tk
from tkinter import ttk, filedialog
# from model_factory import train_model   # 学習機能は mini 版では不要のためコメントアウト
from kizu_kenchi import display_realtime_suiron_with_separate_windows
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Amethyst_mini リアルタイム疵検知")
        self.root.geometry("600x600")
        self.setup_sidebar()
        self.setup_main_frame()
    def setup_sidebar(self):
        self.sidebar = ttk.Frame(self.root, width=200)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        separator = ttk.Separator(self.root, orient="vertical")
        separator.pack(side=tk.LEFT, fill=tk.Y)
        style = ttk.Style()
        style.configure("SidebarButton.TButton", font=("Helvetica", 10))
        style.configure("RealtimeButton.TButton", background="#9E76B4", foreground="black", font=("Helvetica", 10))
        ttk.Label(self.sidebar, text="モードを選択", font=("Helvetica", 12)).pack(pady=10)
        # 以下、学習／ファインチューニングは mini 版で不要
        # self.train_button = ttk.Button(self.sidebar, text="モデル作成 (開発用)",
        #                                command=self.train_mode, style="SidebarButton.TButton")
        # self.train_button.pack(fill=tk.X, padx=10, pady=5, ipadx=10, ipady=5)
        # self.finetune_button = ttk.Button(self.sidebar, text="ファインチューニング",
        #                                   command=self.finetune_mode, style="SidebarButton.TButton")
        # self.finetune_button.pack(fill=tk.X, padx=10, pady=5, ipadx=10, ipady=5)
        self.realtime_button = ttk.Button(
            self.sidebar, text="リアルタイム推論",
            command=self.realtime_mode,
            style="RealtimeButton.TButton"
        )
        self.realtime_button.pack(fill=tk.X, padx=10, pady=5, ipadx=10, ipady=5)
    def setup_main_frame(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        self.default_message = ttk.Label(
            self.main_frame,
            text="サイドバーからモードを選択してください。",
            font=("Helvetica", 12)
        )
        self.default_message.pack(expand=True)
    def clear_main_frame(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()
    # 以下、学習／ファインチューニング機能は mini 版で不要
    # def train_mode(self):
    #     self.clear_main_frame()
    #     …
    # def finetune_mode(self):
    #     self.clear_main_frame()
    #     …
    def realtime_mode(self):
        self.clear_main_frame()
        ttk.Label(self.main_frame, text="リアルタイム疵検知設定", font=("Helvetica", 14)).pack(pady=10)
        # らせん疵モデル (.hef)
        ttk.Label(self.main_frame, text="らせん疵モデル (.hef):").pack(anchor=tk.W, padx=10, pady=5)
        rasen_frame = ttk.Frame(self.main_frame)
        rasen_frame.pack(fill=tk.X, padx=10, pady=5)
        rasen_entry = ttk.Entry(rasen_frame)
        rasen_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(rasen_frame, text="参照", command=lambda: self.select_file(rasen_entry)).pack(side=tk.RIGHT)
        # 黒皮残りモデル (.hef)
        ttk.Label(self.main_frame, text="黒皮残りモデル (.hef):").pack(anchor=tk.W, padx=10, pady=5)
        kuro_frame = ttk.Frame(self.main_frame)
        kuro_frame.pack(fill=tk.X, padx=10, pady=5)
        kuro_entry = ttk.Entry(kuro_frame)
        kuro_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(kuro_frame, text="参照", command=lambda: self.select_file(kuro_entry)).pack(side=tk.RIGHT)
        # 自動保存設定
        auto_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.main_frame, text="自動保存を有効化", variable=auto_var)\
            .pack(anchor=tk.W, padx=10, pady=5)
        # 自動保存 NG率範囲
        ttk.Label(self.main_frame, text="自動保存 NG率範囲 (下限％, 上限％):")\
            .pack(anchor=tk.W, padx=10, pady=5)
        th_frame = ttk.Frame(self.main_frame)
        th_frame.pack(fill=tk.X, padx=10, pady=5)
        ng_lo = ttk.Entry(th_frame, width=6); ng_lo.pack(side=tk.LEFT, padx=5); ng_lo.insert(0,"50")
        ng_hi = ttk.Entry(th_frame, width=6); ng_hi.pack(side=tk.LEFT, padx=5); ng_hi.insert(0,"100")
        # NG率差分閾値 (%)
        ttk.Label(self.main_frame, text="NG率差分閾値 (%):").pack(anchor=tk.W, padx=10, pady=5)
        use_diff = tk.BooleanVar(value=True)
        diff_entry = ttk.Entry(self.main_frame, width=6); diff_entry.insert(0,"2")
        def toggle_diff():
            diff_entry.configure(state=tk.NORMAL if use_diff.get() else tk.DISABLED)
        ttk.Checkbutton(
            self.main_frame,
            text="差分チェックを有効化",
            variable=use_diff,
            command=toggle_diff
        ).pack(anchor=tk.W, padx=10)
        diff_entry.pack(anchor=tk.W, padx=10)
        toggle_diff()
        # 輝度閾値／参照エリア比率
        ttk.Label(self.main_frame, text="輝度閾値:").pack(anchor=tk.W, padx=10, pady=5)
        bright = ttk.Entry(self.main_frame, width=6); bright.pack(anchor=tk.W, padx=10); bright.insert(0,"20")
        ttk.Label(self.main_frame, text="参照エリア比率:").pack(anchor=tk.W, padx=10, pady=5)
        area = ttk.Entry(self.main_frame, width=6); area.pack(anchor=tk.W, padx=10); area.insert(0,"0.1")
        # Start ボタン
        btn = tk.Button(
            self.main_frame,
            text="Start",
            font=("Helvetica", 16),
            command=lambda: self.launch(
                rasen_entry.get().strip(),
                kuro_entry.get().strip(),
                auto_var.get(),
                ng_lo.get(), ng_hi.get(),
                use_diff.get(), diff_entry.get(),
                bright.get(), area.get()
            )
        )
        btn.pack(pady=20)
    def select_file(self, entry):
        path = filedialog.askopenfilename(filetypes=[("HEFファイル", "*.hef")])
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)
    # 以下、学習機能は mini 版で不要
    # def start_training(self, train_dir, model_name):
    #     …
    # def start_finetune(self, train_dir, model_name, pretrained_model):
    #     …
    def launch(self, rasen_p, kuro_p, auto, lo, hi, use_diff, diff, bright, area):
        if not rasen_p or not kuro_p:
            print("モデルファイルを２つとも指定してください。")
            return
        try:
            params = dict(
                auto_save=auto,
                auto_threshold=(int(lo), int(hi)),
                ng_diff_thresh=float(diff),
                use_ng_diff=use_diff,
                brightness=int(bright),
                area_ratio=float(area)
            )
        except ValueError:
            print("数値入力に誤りがあります。")
            return
        display_realtime_suiron_with_separate_windows(
            rasen_p,
            kuro_p,
            auto_save=params["auto_save"],
            auto_save_threshold=params["auto_threshold"],
            cool_time_seconds=1.0,  # 固定値
            ng_rate_diff_threshold=params["ng_diff_thresh"],
            brightness_threshold=params["brightness"],
            area_ratio=params["area_ratio"],
            use_ng_diff=params["use_ng_diff"],
        )
if __name__ == "__main__":
    import sys
    print("Python", sys.version)
    root = tk.Tk()
    App(root)
    root.mainloop()
