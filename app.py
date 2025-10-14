# Camera calibration FAST (OpenCV, sans SB) 
# AUTEUR: Loann KAIKA

import os
import sys
import glob
import time
import threading
import queue
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import scrolledtext as st
from PIL import Image, ImageTk

# =========================
# Réglages "FAST"
# =========================
MAX_VIEWS           = 22          # 12–25 suffit généralement
DETECT_MAX_SIDE     = 1800        # redim max (px) pour la DETECTION (uniquement)
NUM_WORKERS         = max(2, min(8, (os.cpu_count() or 4)))
MIN_AREA_RATIO      = 0.008       # aire(min) damier / aire image (≈0.8%)
USE_FAST_CHECK      = True        # passe rapide puis passe complète
USE_RATIONAL_MODEL  = False       # True -> estime k4..k6 (plus lent)
CLAHE_CLIP          = 3.0
CLAHE_TILE          = (8, 8)

# Affichage (zone fixe du viewer)
VIEW_W              = 1100
VIEW_H              = 680

# Points verts (bien visibles)
POINT_RADIUS        = 8           # rayon du cercle (px)
POINT_THICK         = 3           # épaisseur du tracé

# Zoom du viewer
ZOOM_CHOICES        = ["Fit", "25%", "50%", "75%", "100%", "150%", "200%"]

cv.setUseOptimized(True)
try:
    cv.setNumThreads(NUM_WORKERS)
except Exception:
    pass

# =========================
# Utilitaires généraux
# =========================
def list_images(dir_path):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(dir_path, e)))
    files.sort()
    return files

def cv_to_tk(img_bgr):
    """Convertit BGR -> PhotoImage Tkinter (pas de resize ici : on gère le zoom avant)."""
    if img_bgr is None:
        return None
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    return ImageTk.PhotoImage(pil)

def build_obj_pts(cols, rows, sq_mm):
    n = cols * rows
    obj = np.zeros((n, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj *= float(sq_mm)
    return obj

def per_view_rms(img_pts, prj_pts):
    a = img_pts.reshape(-1, 2).astype(np.float64)
    b = prj_pts.reshape(-1, 2).astype(np.float64)
    if a.shape != b.shape or a.size == 0:
        return np.nan
    d = a - b
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))

def draw_points_green(img_bgr, pts, text=None):
    """Dessine des cercles VERTS bien visibles sur chaque point détecté."""
    out = img_bgr.copy()
    for p in np.asarray(pts).reshape(-1, 2):
        cv.circle(out, (int(p[0]), int(p[1])), POINT_RADIUS, (0, 255, 0), POINT_THICK)
    if text:
        cv.rectangle(out, (10, 10), (560, 72), (0, 0, 0), -1)
        cv.putText(out, text, (18, 52), cv.FONT_HERSHEY_SIMPLEX, 0.9,
                   (255, 255, 255), 2, cv.LINE_AA)
    return out

# =========================
# Détection (robuste, sans SB)
# =========================
def gen_candidates(cols, rows):
    """Génère (cols,rows), inversé, et ±1 autour (≥2)."""
    base = {(cols, rows), (rows, cols)}
    around = [-1, 0, 1]
    cands = set()
    for (c0, r0) in base:
        for dc in around:
            for dr in around:
                c, r = c0 + dc, r0 + dr
                if c >= 2 and r >= 2:
                    cands.add((c, r))
    out = []
    if (cols, rows) in cands:
        out.append((cols, rows)); cands.discard((cols, rows))
    if (rows, cols) in cands:
        out.append((rows, cols)); cands.discard((rows, cols))
    out.extend(sorted(cands))
    return out

def preproc_variants(gray):
    """Plusieurs versions de gray pour améliorer la détection."""
    yield gray
    try:
        clahe = cv.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
        yield clahe.apply(gray)
    except Exception:
        pass
    yield cv.GaussianBlur(gray, (3, 3), 0)
    yield cv.bitwise_not(gray)

def find_corners_multi(gray_full, cand_list):
    """Essaie flags, tailles et prétraitements. Retourne (corners, (cols,rows))."""
    h, w = gray_full.shape[:2]

    # Downscale pour la détection uniquement (pas pour le calibrage)
    scale = 1.0
    if max(h, w) > DETECT_MAX_SIDE:
        scale = DETECT_MAX_SIDE / float(max(h, w))
        gray_small = cv.resize(gray_full, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA)
    else:
        gray_small = gray_full

    flags_fast = (cv.CALIB_CB_ADAPTIVE_THRESH |
                  cv.CALIB_CB_NORMALIZE_IMAGE |
                  (cv.CALIB_CB_FAST_CHECK if USE_FAST_CHECK else 0))
    flags_full = (cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE)

    for flags in (flags_fast, flags_full):
        for (cols, rows) in cand_list:
            for g in preproc_variants(gray_small):
                ok, corners = cv.findChessboardCorners(g, (cols, rows), flags)
                if not ok:
                    continue
                crit = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
                corners = cv.cornerSubPix(g, corners, (11, 11), (-1, -1), crit)

                # Remise à l’échelle dans l’image d’origine
                if scale != 1.0:
                    corners = (corners / scale).astype(np.float32)

                # Filtre par aire couverte (évite les damiers trop petits)
                hull = cv.convexHull(corners.reshape(-1, 2))
                area = float(cv.contourArea(hull)) if hull is not None and len(hull) >= 3 else 0.0
                if area < MIN_AREA_RATIO * (w * h):
                    continue

                return corners, (cols, rows)

    return None, None

def detect_job(path, cand_list):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        gray = cv.imdecode(data, cv.IMREAD_GRAYSCALE)
        if gray is None or gray.size == 0:
            return None
        corners, best_dims = find_corners_multi(gray, cand_list)
        if corners is None:
            return None
        w, h = gray.shape[1], gray.shape[0]
        hull = cv.convexHull(corners.reshape(-1, 2))
        area = float(cv.contourArea(hull)) if hull is not None and len(hull) >= 3 else 0.0
        return (path, corners, (w, h), area, best_dims)
    except Exception:
        return None

def detect_all_parallel(files, cols, rows, log_cb=None):
    cand_list = gen_candidates(cols, rows)
    out = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = [ex.submit(detect_job, p, cand_list) for p in files]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            if r is not None:
                out.append(r)
            if log_cb and (i % 10 == 0):
                log_cb(f"Détection… {i}/{len(futs)}")
    return out

# =========================
# Calibrage principal
# =========================
def calibrate_fast(dir_path, cols, rows, sq_mm, log_cb=None):
    res = {
        "ok": False, "msg": "",
        "K": None, "dist": None, "img_size": None,
        "obj_pts": [], "img_pts": [], "files_used": [],
        "rvecs": [], "tvecs": [], "per_view": [],
        "rms": np.nan, "dt": 0.0, "total": 0, "used": 0,
        "cols": cols, "rows": rows, "sq_mm": float(sq_mm),
        "best_dims": None
    }

    t0 = time.time()
    try:
        files = list_images(dir_path)
        res["total"] = len(files)
        if not files:
            raise RuntimeError("Aucune image trouvée.")

        det = detect_all_parallel(files, cols, rows, log_cb=log_cb)
        if not det:
            raise RuntimeError(
                "Aucun damier détecté. Vérifie :\n"
                "- colsxrows = NOMBRE DE COINS INTÉRIEURS (ex. 10x7 carrés => 9x6 coins)\n"
                "- contraste/NETTETÉ suffisant\n"
                "- c'est bien un damier (pas des cercles)"
            )

        det.sort(key=lambda t: t[3], reverse=True)      # meilleurs d'abord
        sel = det[:MAX_VIEWS]

        dims_counts = Counter([t[4] for t in sel])       # taille majoritaire
        best_dims, _ = dims_counts.most_common(1)[0]
        bcols, brows = best_dims
        res["best_dims"] = best_dims

        obj_one  = build_obj_pts(bcols, brows, sq_mm)
        img_size = None

        for path, corners, (w, h), area, dims in sel:
            if dims != best_dims:
                continue  # on écarte les tailles minoritaires
            if img_size is None:
                img_size = (w, h)
            res["obj_pts"].append(obj_one.copy())
            res["img_pts"].append(corners)
            res["files_used"].append(path)

        res["used"] = len(res["files_used"])
        if res["used"] < 5:
            raise RuntimeError(f"Vues valides insuffisantes ({res['used']}). Il en faut ≥ 5.")

        # ---- Calibration
        K0 = cv.initCameraMatrix2D(res["obj_pts"], res["img_pts"], img_size)

        flags = cv.CALIB_USE_INTRINSIC_GUESS | cv.CALIB_ZERO_TANGENT_DIST
        if USE_RATIONAL_MODEL:
            flags |= cv.CALIB_RATIONAL_MODEL   # calcule k4..k6 (plus lent)

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 60, 1e-6)

        rms, K, dist, rvecs, tvecs = cv.calibrateCamera(
            res["obj_pts"], res["img_pts"], img_size, K0, None,
            flags=flags, criteria=criteria
        )

        res["rms"]      = float(rms)
        res["K"]        = K
        res["dist"]     = dist
        res["rvecs"]    = rvecs
        res["tvecs"]    = tvecs
        res["img_size"] = img_size

        # Erreurs par vue (RMS reprojection)
        for i, p in enumerate(res["files_used"]):
            prj, _ = cv.projectPoints(res["obj_pts"][i], rvecs[i], tvecs[i], K, dist)
            err = per_view_rms(res["img_pts"][i], prj)
            res["per_view"].append({"file": p, "err": float(err)})

        res["ok"] = True

    except Exception as e:
        res["msg"] = f"{e}\n{traceback.format_exc(limit=1)}"
    finally:
        res["dt"] = time.time() - t0

    return res

# =========================
# Interface Tk — toute l’appli scrollable + viewer zoom
# =========================
class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Camera Calibrator (Made by LOANN KAIKA)")
        self.geometry("1280x920")

        # -------- Etat --------
        self.var_dir   = tk.StringVar(value=os.path.abspath("./images_calib"))
        self.var_cols  = tk.StringVar(value="9")
        self.var_rows  = tk.StringVar(value="6")
        self.var_sq    = tk.StringVar(value="25.0")
        self.zoom_var  = tk.StringVar(value="Fit")

        self.res       = None
        self.idx       = 0
        self.tk_img    = None
        self.q         = queue.Queue()
        self.th        = None

        # -------- Conteneur scrollable (toute la page) --------
        self.outer_canvas = tk.Canvas(self, highlightthickness=0)
        self.vbar         = ttk.Scrollbar(self, orient="vertical", command=self.outer_canvas.yview)
        self.outer_canvas.configure(yscrollcommand=self.vbar.set)

        self.vbar.pack(side="right", fill="y")
        self.outer_canvas.pack(side="left", fill="both", expand=True)

        # Frame "page" qui contient TOUT le contenu de l'appli
        self.page = ttk.Frame(self.outer_canvas)
        self.page_window = self.outer_canvas.create_window((0, 0), window=self.page, anchor="nw")

        # Met à jour la scrollregion dès que la page change de taille
        self.page.bind("<Configure>", self._update_scrollregion)
        # Scroll molette (Windows/macOS/Linux)
        self.outer_canvas.bind_all("<MouseWheel>", self._on_mousewheel)   # Windows/macOS
        self.outer_canvas.bind_all("<Button-4>",  lambda e: self.outer_canvas.yview_scroll(-1, "units"))  # Linux
        self.outer_canvas.bind_all("<Button-5>",  lambda e: self.outer_canvas.yview_scroll( 1, "units"))  # Linux

        # Construire l’UI dans la frame "page"
        self._build_ui(self.page)

        # Polling asynchrone
        self.after(100, self._poll_q)

    # ---------- Scroll helpers ----------
    def _update_scrollregion(self, event=None):
        # Ajuste la zone scrollable à la taille totale de la "page"
        self.outer_canvas.configure(scrollregion=self.outer_canvas.bbox("all"))
        # Forcer la largeur de la page = largeur du canvas (évite le scroll horizontal)
        self.outer_canvas.itemconfigure(self.page_window, width=self.outer_canvas.winfo_width())

    def _on_mousewheel(self, event):
        # Windows: event.delta multiple de 120 ; macOS: +/- 1 à 10 ; on normalise
        delta = -1 * int(event.delta / 120) if event.delta else 0
        self.outer_canvas.yview_scroll(delta, "units")

    # ---------- Construction de l'UI ----------
    def _build_ui(self, parent):
        # Paramètres
        top = ttk.LabelFrame(parent, text="Paramètres")
        top.pack(fill="x", padx=10, pady=8)

        ttk.Label(top, text="Dossier :").grid(row=0, column=0, sticky="e", padx=6, pady=6)
        ttk.Entry(top, textvariable=self.var_dir, width=70).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Button(top, text="Parcourir…", command=self._choose_dir).grid(row=0, column=2, padx=6)

        ttk.Label(top, text="Coins intérieurs (cols x rows) :").grid(row=1, column=0, sticky="e", padx=6, pady=6)
        fr = ttk.Frame(top); fr.grid(row=1, column=1, sticky="w")
        ttk.Entry(fr, width=6, textvariable=self.var_cols).pack(side="left")
        ttk.Label(fr, text="x").pack(side="left", padx=4)
        ttk.Entry(fr, width=6, textvariable=self.var_rows).pack(side="left")

        ttk.Label(top, text="Taille carré (mm) :").grid(row=1, column=2, sticky="e", padx=6, pady=6)
        ttk.Entry(top, width=10, textvariable=self.var_sq).grid(row=1, column=3, sticky="w", padx=4)

        self.btn_go = ttk.Button(top, text="Calibrer", command=self._start_calib)
        self.btn_go.grid(row=0, column=4, rowspan=2, padx=10, pady=6, sticky="ns")

        # Progression
        prog = ttk.Frame(parent)
        prog.pack(fill="x", padx=10, pady=(0, 8))
        self.pb = ttk.Progressbar(prog, mode="indeterminate")
        self.pb.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(prog, text="Annuler", command=self._cancel).pack(side="right", padx=6)

        # Résultats (texte avec scroll intégré)
        resf = ttk.LabelFrame(parent, text="Résultats (intrinsèques, distorsion, extrinsèques)")
        resf.pack(fill="x", padx=10, pady=(0, 8))
        self.txt = st.ScrolledText(resf, width=150, height=18, font=("Consolas", 10))
        self.txt.pack(fill="both", expand=True, padx=6, pady=6)
        self._set_text("Prêt. Choisis un dossier, saisis cols/rows (coins INTÉRIEURS) et la taille (mm), puis Calibrer.")

        # Viewer image (zone fixe + scroll interne + ZOOM)
        view = ttk.LabelFrame(parent, text="Images validées (points VERT)")
        view.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        nav = ttk.Frame(view)
        nav.pack(fill="x", padx=6, pady=6)

        self.lbl_idx = ttk.Label(nav, text="Image 0/0")
        self.lbl_idx.pack(side="left")

        # Choix du zoom
        ttk.Label(nav, text="Zoom :").pack(side="right", padx=(6, 3))
        self.zoom_cb = ttk.Combobox(nav, state="readonly",
                                    values=ZOOM_CHOICES, textvariable=self.zoom_var, width=6)
        self.zoom_cb.pack(side="right")
        self.zoom_cb.bind("<<ComboboxSelected>>", lambda e: self._show_img(self.idx))

        ttk.Button(nav, text="← Précédente", command=self._prev).pack(side="right", padx=6)
        ttk.Button(nav, text="Suivante →",  command=self._next).pack(side="right", padx=6)

        # Sous-canvas pour panner de grandes images sans changer la hauteur de page
        viewer = ttk.Frame(view)
        viewer.pack(fill="both", expand=True, padx=6, pady=6)

        self.img_canvas = tk.Canvas(viewer, width=VIEW_W, height=VIEW_H, background="#111")
        xbar = ttk.Scrollbar(viewer, orient="horizontal", command=self.img_canvas.xview)
        ybar = ttk.Scrollbar(viewer, orient="vertical",   command=self.img_canvas.yview)
        self.img_canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)

        self.img_canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        viewer.rowconfigure(0, weight=1)
        viewer.columnconfigure(0, weight=1)

        # Actions
        act = ttk.Frame(parent)
        act.pack(fill="x", padx=10, pady=6)
        ttk.Button(act, text="Enregistrer le rapport (.txt)", command=self._save_report).pack(side="left", padx=4)
        ttk.Button(act, text="Exporter images annotées",       command=self._export_imgs).pack(side="left", padx=4)
        ttk.Button(act, text="Quitter",                        command=self.destroy).pack(side="right", padx=4)

    # ---------- UI helpers ----------
    def _set_text(self, s):
        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", s)
        self.txt.configure(state="disabled")

    def _append_text(self, s):
        self.txt.configure(state="normal")
        self.txt.insert("end", "\n" + s)
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def _choose_dir(self):
        d = filedialog.askdirectory(title="Choisir le dossier d'images")
        if d:
            self.var_dir.set(d)

    def _lock_ui(self, lock=True):
        self.btn_go.configure(state="disabled" if lock else "normal")
        if lock:
            self.pb.start(10)
        else:
            self.pb.stop()

    def _cancel(self):
        self._append_text("Annulation demandée (prise en compte à la prochaine étape).")

    # ---------- Flow principal ----------
    def _start_calib(self):
        try:
            cols = int(self.var_cols.get())
            rows = int(self.var_rows.get())
            if cols < 2 or rows < 2:
                raise ValueError
        except Exception:
            messagebox.showerror("Entrées invalides", "Entrez des entiers ≥ 2 pour cols et rows.")
            return

        try:
            sq_mm = float(self.var_sq.get())
            if sq_mm <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Entrées invalides", "La taille du carré (mm) doit être > 0.")
            return

        dir_path = self.var_dir.get()
        if not os.path.isdir(dir_path):
            messagebox.showerror("Dossier invalide", "Le dossier d'images n'existe pas.")
            return

        self.res    = None
        self.idx    = 0
        self.tk_img = None

        self._lock_ui(True)
        self._set_text("Début de la calibration (FAST)…")

        def log_cb(msg): self.q.put(("log", msg))
        def worker():
            out = calibrate_fast(dir_path, cols, rows, sq_mm, log_cb=log_cb)
            self.q.put(("done", out))

        self.th = threading.Thread(target=worker, daemon=True)
        self.th.start()

    def _poll_q(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._append_text(payload)
                elif kind == "done":
                    self._lock_ui(False)
                    self._handle_res(payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_q)

    # ---------- Résultats + viewer ----------
    def _handle_res(self, res):
        if not res.get("ok", False):
            self._set_text(f"[Alert] Échec :\n{res.get('msg','')}")
            return
        self.res = res
        self._set_text(self._format_res(res))
        self._show_img(0)

    def _format_res(self, r):
        K = r["K"]
        fx, s, cx = K[0, 0], K[0, 1], K[0, 2]
        fy, cy    = K[1, 1], K[1, 2]
        w, h      = r["img_size"]

        # Distorsion : radial/tangentielle
        d = r["dist"].ravel().tolist()
        names = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
        vals  = dict(zip(names, d + [np.nan] * (len(names) - len(d))))
        radial = [vals["k1"], vals["k2"], vals["k3"], vals["k4"], vals["k5"], vals["k6"]]
        tang   = [vals["p1"], vals["p2"]]

        # FOV (degrés) — utile pour le CR du TP
        fovx = float(np.degrees(2 * np.arctan(w / (2.0 * fx))))
        fovy = float(np.degrees(2 * np.arctan(h / (2.0 * fy))))

        def fmt_list(lst):
            return ", ".join(["nan" if (v is None or not np.isfinite(v)) else f"{v:.6e}" for v in lst])

        lines = []
        lines.append("=== Camera Calibration ===")
        lines.append(f"Images utilisées : {r['used']} / {r['total']}")
        lines.append(f"RMS global      : {r['rms']:.4f} px")
        lines.append(f"Image size      : {w} x {h} px")
        lines.append(f"FOVx/FOVy       : {fovx:.2f}°  /  {fovy:.2f}°")
        if r.get("best_dims"):
            lines.append(f"Damier (détecté): {r['best_dims'][0]}x{r['best_dims'][1]} — carré = {r['sq_mm']} mm")
        else:
            lines.append(f"Damier (saisi)  : {r['cols']}x{r['rows']} — carré = {r['sq_mm']} mm")
        lines.append("")
        lines.append("Intrinsic matrix K :")
        lines.append(np.array2string(K, precision=5))
        lines.append("")
        lines.append(f"fx = {fx:.5f} px,  fy = {fy:.5f} px")
        lines.append(f"cx = {cx:.5f} px,  cy = {cy:.5f} px")
        lines.append(f"skew (s) = {s:.6f}")
        lines.append(f"aspect ratio fy/fx = {fy/fx:.6f}")
        lines.append("")
        lines.append("Distorsion radiale      (k1..k6) : [" + fmt_list(radial) + "]")
        lines.append("Distorsion tangentielle (p1, p2) : [" + fmt_list(tang) + "]")
        lines.append("Coefficients complets (OpenCV)    : [" + ", ".join(f"{x:.6e}" for x in r["dist"].ravel()) + "]")
        lines.append("")
        lines.append("Erreurs par image (RMS, px) :")
        for i, pv in enumerate(r["per_view"]):
            lines.append(f"  {i+1:02d}. {os.path.basename(pv['file'])} : {pv['err']:.3f}")
        lines.append("")
        lines.append("Rotation vectors (Rodrigues, radians) et Translation vectors (mm) :")
        for i, (rv, tv) in enumerate(zip(r["rvecs"], r["tvecs"])):
            rv = rv.ravel(); tv = tv.ravel()
            lines.append(f"  View {i+1:02d}  rvec = [{rv[0]:.6f}, {rv[1]:.6f}, {rv[2]:.6f}]"
                         f"   tvec = [{tv[0]:.3f}, {tv[1]:.3f}, {tv[2]:.3f}]")
        lines.append("")
        lines.append(f"Temps total : {r['dt']:.2f} s")
        return "\n".join(lines)

    def _parse_zoom_scale(self, img_w, img_h):
        """Retourne le facteur d'échelle d'affichage en fonction du zoom choisi."""
        z = self.zoom_var.get()
        if z == "Fit":
            # Ajuste pour que l'image tienne dans VIEW_WxVIEW_H (sans agrandir au-delà de 100%)
            s = min(VIEW_W / float(img_w), VIEW_H / float(img_h), 1.0)
            return max(s, 0.05)  # garde un mini
        try:
            if z.endswith("%"):
                p = int(z[:-1])
                return max(p / 100.0, 0.05)
        except Exception:
            pass
        return 1.0

    def _render_overlay(self, idx):
        if not self.res or idx < 0 or idx >= self.res["used"]:
            return None, "Image 0/0"

        path = self.res["files_used"][idx]
        data = np.fromfile(path, dtype=np.uint8)
        img  = cv.imdecode(data, cv.IMREAD_COLOR)
        if img is None:
            return None, f"Impossible de lire : {os.path.basename(path)}"

        out = draw_points_green(img, self.res["img_pts"][idx], text="points détectés (verts)")
        label = (f"Image {idx+1}/{self.res['used']} — "
                 f"{os.path.basename(path)} — "
                 f"RMS(view): {self.res['per_view'][idx]['err']:.3f}px")
        return out, label

    def _show_img(self, idx):
        """Affiche l'image dans le canvas du viewer (avec zoom + scroll interne)."""
        out, label = self._render_overlay(idx)
        self.img_canvas.delete("all")

        if out is None:
            self.lbl_idx.configure(text=label)
            self.tk_img = None
            # scrollregion par défaut
            self.img_canvas.configure(scrollregion=(0, 0, VIEW_W, VIEW_H))
            return

        # ---- Appliquer le ZOOM d'affichage (sans toucher aux calculs)
        h, w = out.shape[:2]
        s = self._parse_zoom_scale(w, h)

        if s != 1.0:
            new_w = max(1, int(w * s))
            new_h = max(1, int(h * s))
            disp  = cv.resize(out, (new_w, new_h), interpolation=cv.INTER_AREA)
        else:
            disp = out

        tkimg = cv_to_tk(disp)
        self.tk_img = tkimg  # garder une référence

        # Place l'image en (0,0) dans le canvas
        self.img_canvas.create_image(0, 0, image=tkimg, anchor="nw")

        # Zone scrollable = taille de l'image affichée
        self.img_canvas.configure(scrollregion=(0, 0, tkimg.width(), tkimg.height()))
        self.lbl_idx.configure(text=label)
        self.idx = idx

        # Met à jour aussi la scrollregion de la page (au cas où)
        self._update_scrollregion()

    def _next(self):
        if not self.res or self.res["used"] == 0:
            return
        self._show_img((self.idx + 1) % self.res["used"])

    def _prev(self):
        if not self.res or self.res["used"] == 0:
            return
        self._show_img((self.idx - 1) % self.res["used"])

    # ---------- Export / Rapport ----------
    def _save_report(self):
        if not self.res:
            messagebox.showinfo("Info", "Pas de résultats.")
            return
        p = filedialog.asksaveasfilename(title="Enregistrer le rapport",
                                         defaultextension=".txt",
                                         filetypes=[("Text files", "*.txt")])
        if not p:
            return
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.write(self._format_res(self.res))
            messagebox.showinfo("OK", f"Rapport enregistré :\n{p}")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    def _export_imgs(self):
        if not self.res:
            messagebox.showinfo("Info", "Calibrez d'abord.")
            return
        d = filedialog.askdirectory(title="Choisir le dossier d'export")
        if not d:
            return
        ok = 0
        for i, path in enumerate(self.res["files_used"]):
            data = np.fromfile(path, dtype=np.uint8)
            img  = cv.imdecode(data, cv.IMREAD_COLOR)
            if img is None:
                continue
            out = draw_points_green(img, self.res["img_pts"][i], text="points detected (green)")
            base = os.path.splitext(os.path.basename(path))[0]
            outp = os.path.join(d, base + "_annotated.jpg")
            try:
                cv.imencode(".jpg", out)[1].tofile(outp)
                ok += 1
            except Exception:
                pass
        messagebox.showinfo("Export", f"{ok}/{self.res['used']} images annotées exportées.")

if __name__ == "__main__":
    App().mainloop()
