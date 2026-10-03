"""
Extraction, calibration and digitisation of the raster figures of

    McEllistrim, Garcia-Ruiz, Goodwin, Fal'ko, "Spectroscopic signatures of
    tetralayer graphene polytypes", arXiv:2302.07374v1 (2023).

The figures are embedded in the PDF as raster images (Fig. 1: 3844 x 1016 px,
Fig. 2: 1065 x 992 px, Fig. 7: 832 x 992 px).  This module locates the axes of
the band-structure, ARPES and density-of-states panels and returns pixel ->
data mappings, and digitises the DOS curves (blue lines).

Requires PyMuPDF (``pip install pymupdf``) for the extraction and Pillow.
"""
import os

import numpy as np
from PIL import Image

PAGE_INDEX = {"fig1": 1, "fig2": 3, "fig7": 10}      # zero-based PDF pages


# =============================================================================
# extraction
# =============================================================================

def extract_figures(pdf_path, out_dir):
    """Save the raster of Figs. 1, 2 and 7 as PNG files; return {name: path}."""
    import pymupdf  # noqa: WPS433 (optional dependency)
    os.makedirs(out_dir, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    paths = {}
    for name, pno in PAGE_INDEX.items():
        images = doc[pno].get_images()
        if not images:
            raise RuntimeError(f"no raster image on page {pno + 1}")
        pix = pymupdf.Pixmap(doc, max(images, key=lambda im: im[2] * im[3])[0])
        if pix.n > 3:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        path = os.path.join(out_dir, f"{name}.png")
        pix.save(path)
        paths[name] = path
    return paths


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


# =============================================================================
# pixel classification and small helpers
# =============================================================================

def dark_mask(arr, thr=175):
    """Dark, non-blue pixels (frames, ticks, text)."""
    lum = arr.mean(axis=-1)
    bluish = (arr[..., 2].astype(int) - arr[..., 0].astype(int)) > 40
    return (lum < thr) & ~bluish


def black_mask(arr, thr=70):
    a = arr.astype(int)
    return (a[..., 0] < thr) & (a[..., 1] < thr) & (a[..., 2] < thr)


def blue_mask(arr):
    """The blue DOS curves of the paper."""
    a = arr.astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (b > 140) & (r < 130) & (g < 130) & (b - r > 60)


def clusters(idx, gap=1):
    """Group sorted integer indices into runs separated by more than `gap`."""
    idx = np.asarray(idx)
    if idx.size == 0:
        return []
    groups = [[idx[0]]]
    for i in idx[1:]:
        if i - groups[-1][-1] <= gap:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def best_progression(cand, n, tol=3.0, min_step=5.0):
    """Pick n candidates forming the most regular arithmetic progression."""
    cand = np.asarray(cand, dtype=float)
    if cand.size < n:
        return None
    if cand.size == n:
        return cand
    best = None
    for i in range(cand.size):
        for j in range(i + 1, cand.size):
            for m in range(1, n):
                s = (cand[j] - cand[i]) / m
                if s < min_step:
                    continue
                k = np.round((cand - cand[i]) / s)
                ok = np.abs(cand - (cand[i] + k * s)) < tol
                uniq = {}
                for c_, k_ in zip(cand[ok], k[ok]):
                    uniq.setdefault(int(k_), c_)
                keys = sorted(uniq)
                for start in range(len(keys) - n + 1):
                    run = keys[start:start + n]
                    if run[-1] - run[0] == n - 1:
                        pts = np.array([uniq[kk] for kk in run])
                        score = np.abs(np.diff(pts) - s).sum()
                        if best is None or score < best[0]:
                            best = (score, pts)
    return None if best is None else best[1]


def linear_map(pix, vals):
    """Least-squares linear map pixel -> value; returns (function, residuals)."""
    pix = np.asarray(pix, dtype=float)
    vals = np.asarray(vals, dtype=float)
    A = np.vstack([pix, np.ones_like(pix)]).T
    coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
    return (lambda p: coef[0] * np.asarray(p, dtype=float) + coef[1]), A @ coef - vals


# =============================================================================
# generic DOS-panel tools
# =============================================================================

def find_frame(arr, frac_v=0.6, frac_h=0.6, thr=175):
    """Axes frame of a DOS panel: (x_left, x_right, y_top, y_bottom) in `arr`."""
    d = dark_mask(arr, thr)
    H, W = d.shape
    cols = np.nonzero(d.sum(axis=0) > frac_v * H)[0]
    rows = np.nonzero(d.sum(axis=1) > frac_h * W)[0]
    if cols.size < 2 or rows.size < 2:
        raise RuntimeError(f"frame not found (cols={cols}, rows={rows})")
    cc = [int(round(np.mean(g))) for g in clusters(cols, 2)]
    rc = [int(round(np.mean(g))) for g in clusters(rows, 2)]
    return cc[0], cc[-1], rc[0], rc[-1]


def find_ticks(arr, x0, x1, y0, y1, side, n_expected=None, near=(3, 11), far=(14, 34), min_dark=4):
    """Tick marks inside a frame edge: dark within `near` px of the edge, none within `far` px."""
    d = dark_mask(arr)
    if side == "right":
        near_strip = d[y0 + 3:y1 - 2, x1 - near[1]:x1 - near[0]]
        far_strip = d[y0 + 3:y1 - 2, x1 - far[1]:x1 - far[0]]
        offset = y0 + 3
    elif side == "left":
        near_strip = d[y0 + 3:y1 - 2, x0 + near[0]:x0 + near[1]]
        far_strip = d[y0 + 3:y1 - 2, x0 + far[0]:x0 + far[1]]
        offset = y0 + 3
    elif side == "bottom":
        near_strip = d[y1 - near[1]:y1 - near[0], x0 + 3:x1 - 2].T
        far_strip = d[y1 - far[1]:y1 - far[0], x0 + 3:x1 - 2].T
        offset = x0 + 3
    else:
        near_strip = d[y0 + near[0]:y0 + near[1], x0 + 3:x1 - 2].T
        far_strip = d[y0 + far[0]:y0 + far[1], x0 + 3:x1 - 2].T
        offset = x0 + 3
    good = (near_strip.sum(axis=1) >= min_dark) & (far_strip.sum(axis=1) == 0)
    cand = np.array([np.mean(g) for g in clusters(np.nonzero(good)[0] + offset, 2)])
    if n_expected is None or cand.size <= n_expected:
        return cand
    sel = best_progression(cand, n_expected)
    return cand if sel is None else sel


def digitize_curve(arr, x0, x1, y0, y1, ymap, xmap):
    """Per pixel row inside the frame, the rightmost run of blue pixels.

    Returns an array of rows (E, dos_mean, dos_min, dos_max) sorted by E."""
    b = blue_mask(arr)
    out = []
    for y in range(y0 + 2, y1 - 1):
        xs = np.nonzero(b[y, x0 + 2:x1 - 1])[0] + x0 + 2
        if xs.size == 0:
            continue
        run = clusters(xs, 2)[-1]
        out.append((ymap(y), xmap(np.mean(run)), xmap(min(run)), xmap(max(run))))
    data = np.array(out)
    return data[np.argsort(data[:, 0])]


# =============================================================================
# Fig. 1: tetralayer band panels (3D view) and DOS panels
# =============================================================================

FIG1_STACKS = ["ABAB", "ABCA", "ABCB"]


def fig1_dos_panels(img):
    """Calibrate and digitise the three DOS panels of Fig. 1.

    Each entry: name, frame (x0, x1, y0, y1) in absolute pixels, ymap (row ->
    eV), xmap (column -> states/(eV cm^2)), data (digitised curve)."""
    H, W, _ = img.shape
    yvals = np.arange(0.6, -0.61, -0.2)
    panels = []
    for i, name in enumerate(FIG1_STACKS):
        xa = int(W * i / 3) + int(W / 3 * 0.45)
        xb = int(W * (i + 1) / 3)
        sub = img[:, xa:xb]
        x0, x1, y0, y1 = find_frame(sub)
        yt = find_ticks(sub, x0, x1, y0, y1, "right", n_expected=7)
        if len(yt) != 7:
            yt = find_ticks(sub, x0, x1, y0, y1, "right", n_expected=7, far=(14, 20))
        if len(yt) != 7:
            raise RuntimeError(f"Fig. 1 {name}: found {len(yt)} energy ticks")
        ymap, _ = linear_map(yt, yvals)
        xmap, _ = linear_map([x0, x1], [0.0, 5e14])        # frame = axis limits 0..5e14
        data = digitize_curve(sub, x0, x1, y0, y1, ymap, xmap)
        xmap_abs = (lambda x, xmap=xmap, xa=xa: xmap(np.asarray(x, dtype=float) - xa))
        panels.append({"name": name, "frame": (x0 + xa, x1 + xa, y0, y1),
                       "ymap": ymap, "xmap": xmap_abs, "data": data})
    return panels


def fig1_band_panels(img):
    """Calibrate the 3D band-structure panels of Fig. 1 (view azimuth 0).

    The vertical axis through the origin carries ticks at E/gamma1 = -2..2 and
    the p_x axis on the grey plane carries ticks at p_x/p_c = -3..3, so for the
    p_y = 0 cut: column = axis_col + px_per_pc * p_x/p_c,
    row = row0 - px_per_gamma1 * E/gamma1."""
    H, W, _ = img.shape
    blk = black_mask(img)

    def run_len(row, c):
        if not row[c]:
            return 0
        l = c
        while l > 0 and row[l - 1]:
            l -= 1
        r = c
        while r < len(row) - 1 and row[r + 1]:
            r += 1
        return r - l + 1

    panels = []
    for i, name in enumerate(FIG1_STACKS):
        xa = int(W * i / 3)
        xb = xa + int(W / 3 * 0.45)
        sub = blk[:, xa:xb]
        c = int(np.argmax(sub.sum(axis=0)))
        lengths = np.array([run_len(sub[y], c) for y in range(H)])
        groups = clusters(np.nonzero((lengths >= 9) & (lengths <= 40))[0], 2)
        rows = np.array([np.mean(g) for g in groups])
        # ticks at +2 and -2 are the outermost; +1/-1 are 1/4 and 3/4 of the way
        if rows.size < 2:
            raise RuntimeError(f"Fig. 1 {name}: energy ticks not found")
        top, bottom = rows.min(), rows.max()
        px_per_gamma1 = (bottom - top) / 4.0
        row0 = 0.5 * (top + bottom)
        inner = rows[(rows > top + 1) & (rows < bottom - 1)]
        expected = row0 + px_per_gamma1 * np.array([-1.0, 1.0])
        ok = all(np.abs(inner - e).min() < 6 for e in expected) if inner.size else True
        if not ok:
            raise RuntimeError(f"Fig. 1 {name}: energy ticks irregular: {rows}")
        # p_x ticks: short black marks across the plane axis line near row0
        yl = int(round(row0))
        band = sub[yl - 14:yl + 14]
        prof = band.sum(axis=0)
        cand = np.nonzero(prof >= 5)[0]
        tick_cols = np.array([np.mean(g) for g in clusters(cand, 3) if len(g) <= 8])
        tick_cols = tick_cols[np.abs(tick_cols - c) > 20]            # exclude the axis itself
        steps = np.abs(tick_cols - c)
        px_per_pc = np.median(steps / np.round(steps / np.median(np.diff(np.sort(tick_cols)))))
        panels.append({"name": name, "axis_col": c + xa, "row0": row0,
                       "px_per_gamma1": px_per_gamma1, "px_per_pc": float(px_per_pc),
                       "crop": (xa, min(xb + 60, int(W * (i + 1) / 3)), 0, H)})
    return panels


# =============================================================================
# Fig. 2: ARPES constant-momentum cuts through K along p_y (row 1)
# =============================================================================

FIG2_PANELS = [("ABAB", 99, 249), ("ABCA", 249, 398), ("ABCB", 398, 548), ("ABAC", 548, 698),
               ("ABA", 757, 907), ("ABC", 907, 1057)]


def fig2_arpes_panels(img):
    """Calibrate the six constant-momentum cuts of Fig. 2 (omega = 70 eV row).

    Energy from the tick labels left of the first panel, momentum from the
    0.1 1/A scale bar, K at the mirror-symmetry column of each panel."""
    lum = img.mean(axis=2)
    blk = lum < 110
    # frame rows of the first row of cuts
    rows = np.nonzero(blk[:, 100:700].sum(axis=1) > 400)[0]
    rows = [int(round(np.mean(g))) for g in clusters(rows, 2)]
    y0, y1 = rows[1], rows[2]
    # energy labels "0", "-0.5", "-1" left of the first panel
    strip = blk[y0 - 5:y1 + 5, 78:97]
    lab = [np.mean(g) + y0 - 5 for g in clusters(np.nonzero(strip.sum(axis=1) >= 1)[0], 1)]
    if len(lab) != 3:
        raise RuntimeError(f"Fig. 2: expected 3 energy labels, found {lab}")
    emap, _ = linear_map(lab, [0.0, -0.5, -1.0])
    panels = []
    for name, x0, x1 in FIG2_PANELS:
        # scale bar: horizontal black run of 15..80 px near the bottom left
        bar = None
        for y in range(y1 - 50, y1 - 1):
            for g in clusters(np.nonzero(blk[y, x0 + 2:x1 - 2])[0], 1):
                if 15 <= len(g) <= 80 and (g[0] + x0 + 2) < (x0 + x1) / 2:
                    bar = len(g)
                    break
            if bar:
                break
        if bar is None:
            bar = 29
        px_per_invA = bar / 0.1
        # mirror symmetry column
        I = 255.0 - lum[y0 + 3:y1 - 3, x0 + 3:x1 - 3]
        w = I.shape[1]
        best = None
        for cc in range(w // 2 - 15, w // 2 + 15):
            half = min(cc, w - 1 - cc)
            a = I[:, cc - half:cc + half + 1]
            s = ((a - a[:, ::-1]) ** 2).mean()
            if best is None or s < best[0]:
                best = (s, cc + x0 + 3)
        panels.append({"name": name, "frame": (x0, x1, y0, y1), "emap": emap,
                       "k_col": best[1], "px_per_invA": px_per_invA})
    return panels


# =============================================================================
# Fig. 7: trilayer and pentalayer panels
# =============================================================================

FIG7_LAYOUT = {   # name: (row, col, energy tick values, DOS tick values)
    "ABC": (0, 0, np.arange(0.6, -0.61, -0.2), np.array([0, 1, 2, 3]) * 1e14),
    "ABA": (0, 1, np.arange(0.6, -0.61, -0.2), np.array([0, 5, 10, 15]) * 1e13),
    "ABCAB": (1, 0, np.arange(0.8, -0.81, -0.4), np.array([0, 2, 4, 6]) * 1e14),
    "ABACA": (1, 1, np.arange(0.8, -0.81, -0.4), np.array([0, 2, 4, 6]) * 1e14),
    "ABCAC": (2, 0, np.arange(0.8, -0.81, -0.4), np.array([0, 2, 4, 6]) * 1e14),
    "ABABC": (2, 1, np.arange(0.8, -0.81, -0.4), np.array([0, 2, 4, 6]) * 1e14),
    "ABCBA": (3, 0, np.arange(0.8, -0.81, -0.4), np.array([0, 2, 4, 6]) * 1e14),
    "ABABA": (3, 1, np.arange(0.8, -0.81, -0.4), np.array([0, 2, 4, 6]) * 1e14),
}


def fig7_cells(img):
    """Pixel boxes (xa, xb, ya, yb) of the eight panels (band plot + DOS plot)."""
    H, W, _ = img.shape
    cells = {}
    for name, (r, c, _, _) in FIG7_LAYOUT.items():
        cells[name] = (int(W * c / 2), int(W * (c + 1) / 2), int(H * r / 4), int(H * (r + 1) / 4))
    return cells


def fig7_dos_panels(img):
    """Calibrate (from the tick-label text) and digitise the DOS panels of Fig. 7."""
    H, W, _ = img.shape
    panels = []
    for name, (r, c, yvals, xvals) in FIG7_LAYOUT.items():
        ya, yb = int(H * r / 4), int(H * (r + 1) / 4)
        xa = int(W * c / 2) + int(W / 2 * 0.5)
        xb = int(W * (c + 1) / 2)
        sub = img[ya:yb, xa:xb]
        try:
            x0, x1, y0, y1 = find_frame(sub, 0.4, 0.4, thr=215)
        except RuntimeError:
            d = dark_mask(sub, 215)
            cols = np.nonzero(d.sum(axis=0) > 0.4 * d.shape[0])[0]
            rows = np.nonzero(d.sum(axis=1) > 0.4 * d.shape[1])[0]
            x0, y0, y1 = int(cols[0]), int(rows[0]), int(rows[-1])
            x1 = x0 + 138
        strip = dark_mask(sub[:, x1 + 3:x1 + 30], 200)
        groups = [g for g in clusters(np.nonzero(strip.sum(axis=1) >= 2)[0], 1)
                  if 4 <= len(g) <= 12 and y0 - 6 <= np.mean(g) <= y1 + 6]
        yt = best_progression([np.mean(g) for g in groups], len(yvals))
        stripx = dark_mask(sub[y1 + 3:y1 + 13, max(0, x0 - 10):x1 + 12], 200)
        cols = np.nonzero(stripx.sum(axis=0) >= 2)[0] + max(0, x0 - 10)
        gx = [g for g in clusters(cols, 8) if 2 <= len(g) <= 20]
        xt = best_progression([np.mean(g) for g in gx], len(xvals))
        if yt is None or xt is None:
            raise RuntimeError(f"Fig. 7 {name}: tick labels not found")
        ymap, _ = linear_map(yt, yvals)
        xmap, _ = linear_map(xt, xvals)
        data = digitize_curve(sub, x0, x1, y0, y1, ymap, xmap)
        panels.append({"name": name, "frame": (x0 + xa, x1 + xa, y0 + ya, y1 + ya),
                       "ymap": (lambda y, ymap=ymap, ya=ya: ymap(np.asarray(y, dtype=float) - ya)),
                       "xmap": (lambda x, xmap=xmap, xa=xa: xmap(np.asarray(x, dtype=float) - xa)),
                       "data": data})
    return panels


def write_digitised_csv(path, data, header_lines):
    with open(path, "w") as fh:
        for line in header_lines:
            fh.write(f"# {line}\n")
        fh.write("energy_eV,dos_cm2eV_mean,dos_cm2eV_min,dos_cm2eV_max\n")
        for row in data:
            fh.write(",".join(f"{v:.10g}" for v in row) + "\n")


def read_digitised_csv(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("energy"):
                continue
            rows.append([float(v) for v in line.split(",")])
    return np.array(rows)
