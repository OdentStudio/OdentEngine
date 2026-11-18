"""odent_engine smoke test: runs the built engine (bin/<platform>/) through the add-on's
client (client/engine.py, a copy of the add-on's) on a small synthetic CT, outside Blender.

    python tests/smoke_test.py [DICOM folder ...]

Any Python 3.9+ with numpy works; on a computer with only Blender, use Blender's:
    <blender>/5.2/python/bin/python.exe tests/smoke_test.py   (Windows)

It checks what differs between platforms: starting the engine, the shared memory
both ways (engine -> addon for volumes, addon -> engine for the cut's mesh),
file names with non-Latin letters (a Cyrillic folder), .mha writing and reading,
marching cubes, the slices, the loop cut, the Load Data thumbnails and the registration
(register, the AI align's: a known move recovered). With DICOM folders given, it also
scans and reads their largest series. Prints one line per check and exits 1 when
one fails.
"""

import importlib.util
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# the add-on's engine client: client/engine.py (its copy here), or the add-on's own
# (ODENT_ENGINE_CLIENT=<add-on>/addon/core/engine.py: that add-on's bin/ engine is tested)
ENGINE_PY = os.environ.get("ODENT_ENGINE_CLIENT") or os.path.join(HERE, "..", "client", "engine.py")

spec = importlib.util.spec_from_file_location("odent_engine_client", ENGINE_PY)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)

failed = []


def check(name, ok, info=""):
    print(f"{'OK  ' if ok else 'FAIL'} {name}{'  ' + info if info else ''}")
    if not ok:
        failed.append(name)


def run(name, fn):
    try:
        fn()
    except Exception as er:  # noqa: BLE001
        check(name, False, f"{type(er).__name__}: {er}")


def write_mha(path, a, spacing):
    """A minimal MetaImage of an int16 array (z, y, x), raw (not compressed)."""
    header = (
        "ObjectType = Image\nNDims = 3\nBinaryData = True\nBinaryDataByteOrderMSB = False\n"
        "CompressedData = False\nTransformMatrix = 1 0 0 0 1 0 0 0 1\nOffset = 0 0 0\n"
        f"ElementSpacing = {spacing} {spacing} {spacing}\n"
        f"DimSize = {a.shape[2]} {a.shape[1]} {a.shape[0]}\nElementType = MET_SHORT\n"
        "ElementDataFile = LOCAL\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.ascontiguousarray(a, dtype="<i2").tobytes())


# a 64^3 CT: air, a "bone" ball of 1000 HU, radius 20 voxels
n, sp = 64, 0.5
z, y, x = np.mgrid[:n, :n, :n]
ball = ((x - 32) ** 2 + (y - 32) ** 2 + (z - 32) ** 2) <= 20 ** 2
ct = np.where(ball, 1000, -1000).astype(np.int16)

exe = client.executable()
check("engine binary present", os.path.isfile(exe), exe)
if failed:
    sys.exit(1)
eng = client.Engine()

tmp = tempfile.mkdtemp(prefix="odent_smoke_")
folder = os.path.join(tmp, "Пациент Тестовый 11 21")   # Cyrillic and spaces, as in real cases (an invented name)
os.makedirs(folder)
src = os.path.join(folder, "ct_исходный.mha")
write_mha(src, ct, sp)


def t_ping():
    t = time.perf_counter()
    eng.start()
    check("engine starts, protocol version", eng.info.get("api") == client.API,
          f"{eng.info} ({1000 * (time.perf_counter() - t):.0f} ms)")
    with open(os.path.join(HERE, "..", "VERSION")) as f:
        version = f.read().strip()
    check("engine version is VERSION's", eng.info.get("version") == version,
          f"{eng.info.get('version')} / {version}")


def t_load():
    info = eng.call("image.load", file=src)
    a = eng.array(info)
    check("image.load of a Cyrillic path, voxels by shared memory",
          a.shape == ct.shape and np.array_equal(a, ct), f"{a.shape} {a.dtype}")
    check("geometry", np.allclose(info["spacing"], sp) and np.allclose(info["origin"], 0),
          f"spacing {info['spacing']}")
    out = os.path.join(folder, "копия.mha")
    # (metadata values stay ASCII: ITK's MetaImage writer drops other letters; the
    # addon writes only ASCII ones, the series UID and its own tags)
    eng.call("image.write", volume=info["volume"], path=out, metadata={"odent_test": "1.2.840.5"})
    back = eng.call("image.load", file=out)
    check("image.write + read back (Cyrillic name, metadata)",
          np.array_equal(eng.array(back), ct) and back.get("metadata", {}).get("odent_test") == "1.2.840.5",
          str(back.get("metadata", {}).get("odent_test")))
    eng.free(back)
    h = eng.call("image.header", file=out)
    check("image.header", list(h["size"]) == [n, n, n], str(h["size"]))
    iso = eng.call("image.resample_iso", volume=info["volume"], spacing=1.0, fill=-1000.0)
    check("image.resample_iso", tuple(eng.array(iso).shape) == (32, 32, 32), str(eng.array(iso).shape))
    eng.free(iso)

    m = eng.call("mesh.threshold", volume=info["volume"], threshold=0.0)
    v = eng.array(m["vertices"]).reshape(-1, 3)
    r = np.linalg.norm(v - 32 * sp, axis=1)
    check("mesh.threshold (marching cubes)", len(v) > 1000 and abs(r.mean() - 20 * sp) < 0.3,
          f"{len(v)} vertices, radius {r.mean():.2f} mm (ball {20 * sp} mm)")
    eng.free(m["vertices"])
    eng.free(m["faces"])

    c = 32 * sp
    plane = {"origin": [c - 15 * sp, c - 15 * sp, c], "direction": [1, 0, 0, 0, 1, 0, 0, 0, 1],
             "size": [31, 31, 1], "spacing": [sp, sp, 1.0]}
    tiles = eng.call("slices.compute", volume=info["volume"], fill=-1000.0, planes=[plane])["tiles"]
    s = eng.array(tiles[0])[0]
    check("slices.compute", s.shape == (31, 31) and s[15, 15] == 1000 and s[0, 0] == -1000,
          f"{s.shape} centre {s[15, 15]} corner {s[0, 0]}")
    eng.free(tiles[0])
    eng.free(info)


def t_cut():
    # a 41 x 41 grid of quads, 20 mm, cut by a circle of 6 mm
    k = 41
    gx, gy = np.meshgrid(np.linspace(-10, 10, k), np.linspace(-10, 10, k))
    co = np.c_[gx.ravel(), gy.ravel(), np.zeros(k * k)]
    quads = [(j * k + i, j * k + i + 1, (j + 1) * k + i + 1, (j + 1) * k + i)
             for j in range(k - 1) for i in range(k - 1)]
    conn = np.array(quads, dtype=np.int64).ravel()
    offsets = np.arange(0, len(conn) + 1, 4, dtype=np.int64)
    blocks = []

    def put(values, dtype):
        a = np.ascontiguousarray(values, dtype=dtype)
        a = a.reshape(len(a), -1)
        info = eng.call("buffer.new", dtype=dtype, rows=int(a.shape[0]), cols=int(a.shape[1]))
        blocks.append(info)
        eng.array(info)[0][...] = a
        return info["volume"]

    loop = [[6 * np.cos(t), 6 * np.sin(t), 0.0] for t in np.linspace(0, 2 * np.pi, 60, endpoint=False)]
    res = eng.call("mesh.cut_loop", co=put(co, "float64"), offsets=put(offsets, "int64"),
                   conn=put(conn, "int64"), point={"height": put(co[:, 2:], "float64")}, face={},
                   loop=loop, smallest=True)
    inside = res.get("inside")
    pts = eng.array(inside["co"])[0] if inside else np.zeros((0, 3))
    rad = np.linalg.norm(pts[:, :2], axis=1) if len(pts) else np.array([99.0])
    check("mesh.cut_loop (mesh to the engine by shared memory)",
          inside is not None and res.get("outside") is not None and rad.max() < 6.6,
          f"inside {len(pts)} points, farthest {rad.max():.2f} mm from the centre (loop 6 mm)")
    for key in ("inside", "outside"):
        r = res.get(key) or {}
        for name in ("co", "offsets", "conn"):
            if name in r:
                blocks.append(r[name])
        for d in ("point", "face"):
            blocks.extend((r.get(d) or {}).values())
    for info in blocks:
        eng.free(info)


def t_thumbnails():
    # the Load Data dialog's previews: the CT's middle axial slice, a scan file seen along
    # its thinnest axis (a 30 x 20 x 5 mm box as binary STL, ASCII PLY and OBJ)
    # the coronal maximum intensity projection, against numpy's; this CT has a bar of
    # 1500 HU (brighter than the ball) high up (z indexes 44-52): it comes out near
    # the top, and at the bottom when the slice spacing is negative (a multi-frame
    # DICOM can give one: its frames run the other way)
    marked = ct.copy()
    marked[44:52, 1:5, 26:38] = 1500
    src_marked = os.path.join(folder, "ct_bar.mha")
    write_mha(src_marked, marked, sp)
    tile = eng.call("image.thumbnail", file=src_marked, size=130)
    a = eng.array(tile)[0]
    (c0, cw), (s0, sw) = tile["panels"]
    cor, sag = a[:, c0:c0 + cw], a[:, s0:s0 + sw]
    want = [marked.max(axis=1)[::-1], marked.max(axis=2)[::-1, ::-1]]   # head up; sagittal: front right
    # flip: x runs with the picture, y against it (index 0, the front, on the right), z with it
    same = [p.shape == w.shape and np.allclose(p, w) for p, w in zip((cor, sag), want)]
    bar = np.nonzero((cor >= 1400).any(axis=1))[0]
    check("image.thumbnail (coronal + sagittal maximum intensity projections)",
          all(same) and len(bar) and bar.mean() < n / 2 and np.isnan(a[:, c0 + cw:s0]).all()
          and tile["flip"] == [False, True, False] and np.allclose(tile["extent_mm"], [n * sp] * 3),
          f"{a.shape}, panels = numpy's MIPs: {same}, bar in rows {bar.min()}..{bar.max()} (top), "
          f"flip {tile['flip']}, {tile['slices_read']} slices read")
    eng.free(tile)
    neg = os.path.join(folder, "ct_negative_z.mha")
    with open(src_marked, "rb") as f:
        raw = f.read()
    with open(neg, "wb") as f:
        f.write(raw.replace(f"ElementSpacing = {sp} {sp} {sp}".encode(), f"ElementSpacing = {sp} {sp} -{sp}".encode(), 1))
    tile = eng.call("image.thumbnail", file=neg, size=130)
    a = eng.array(tile)[0]
    c0, cw = tile["panels"][0]
    bar = np.nonzero((a[:, c0:c0 + cw] >= 1400).any(axis=1))[0]
    check("image.thumbnail, negative slice spacing",
          a.shape[0] == n and len(bar) and bar.mean() > n / 2,
          f"{a.shape}, bar in rows {bar.min() if len(bar) else '-'}..{bar.max() if len(bar) else '-'} (bottom)")
    eng.free(tile)
    v = np.array([[x, y, z] for z in (0, 5) for y in (0, 20) for x in (0, 30)], dtype=np.float32)
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    tris = [(q[0], q[1], q[2]) for q in quads] + [(q[0], q[2], q[3]) for q in quads]
    stl = os.path.join(folder, "верх_UpperJaw.stl")
    with open(stl, "wb") as f:
        f.write(b"\0" * 80 + np.uint32(len(tris)).tobytes())
        for t in tris:
            f.write(np.zeros(3, np.float32).tobytes() + v[list(t)].tobytes() + b"\0\0")
    ply = os.path.join(folder, "box.ply")
    with open(ply, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex 8\nproperty float x\nproperty float y\n"
                f"property float z\nproperty uchar red\nelement face 6\n"
                f"property list uchar int vertex_indices\nend_header\n")
        f.writelines(f"{x} {y} {z} 200\n" for x, y, z in v)
        f.writelines(f"4 {' '.join(map(str, q))}\n" for q in quads)
    obj = os.path.join(folder, "box.obj")
    with open(obj, "w") as f:
        f.writelines(f"v {x} {y} {z}\n" for x, y, z in v)
        f.writelines(f"f {' '.join(f'{i + 1}/1/1' for i in q)}\n" for q in quads)
    for path in (stl, ply, obj):
        m = eng.call("mesh.thumbnail", file=path, size=64)
        img = eng.array(m)[0]
        ext = sorted(m["size_mm"], reverse=True)
        check(f"mesh.thumbnail {os.path.splitext(path)[1]}",
              m["triangles"] == 12 and np.allclose(ext, [30, 20, 5], atol=1e-3)
              and img.shape == (64, 64) and (img > 0).mean() > 0.4 and img[0, 0] == -1,
              f"{m['triangles']} triangles, {[round(e, 2) for e in ext]} mm, "
              f"{100 * (img > 0).mean():.0f} % drawn")
        eng.free(m)


def t_crop():
    # Load Data's VOI: a box of the CT, at the same place in space
    info = eng.call("image.load", file=src)
    c = eng.call("image.crop", volume=info["volume"], lo=[10, 20, 30], hi=[50, 44, 60])
    got = eng.array(c)
    want = ct[30:60, 20:44, 10:50]   # numpy: z, y, x
    check("image.crop (VOI)",
          got.shape == want.shape and np.array_equal(got, want)
          and np.allclose(c["origin"], [10 * sp, 20 * sp, 30 * sp]),
          f"{got.shape}, origin {c['origin']} (the first voxel kept)")
    eng.free(c)
    eng.free(info)


def t_dicom(folder):
    t = time.perf_counter()
    series = eng.call("dicom.scan", dir=folder)["series"]
    ok = [s for s in series if not s.get("error")]
    check(f"dicom.scan {os.path.basename(folder)}", bool(ok),
          f"{len(series)} series ({time.perf_counter() - t:.1f} s)")
    if not ok:
        return
    best = max(ok, key=lambda s: s["count"])
    check("dicom.scan details (Load Data)", all(k in best for k in ("description", "date", "modality")),
          f"{best.get('modality')} {best.get('date')}")
    t = time.perf_counter()
    tile = eng.call("image.thumbnail", dir=folder, series=best["id"], size=256)
    a = eng.array(tile)[0]
    check("image.thumbnail of the series",
          a.ndim == 2 and a.shape[0] > 8 and float(np.nanmax(a)) > float(np.nanmin(a)),   # NaN around the panels
          f"{a.shape} ({1000 * (time.perf_counter() - t):.0f} ms)")
    eng.free(tile)
    t = time.perf_counter()
    info = eng.call("image.load", dir=folder, series=best["id"])
    a = eng.array(info)
    check(f"image.load series ({best['count']} files)", a.ndim == 3 and a.size > 0,
          f"{a.shape} {a.dtype} {info['pixel_type']} ({time.perf_counter() - t:.1f} s)")
    eng.free(info)


def _bumpy_surface(step=0.25):
    """An uneven 40 x 30 mm patch, cusps off-centre (no symmetry): tools/meshregauto's test."""
    gx, gy = np.meshgrid(np.arange(-20.0, 20.0 + 1e-9, step), np.arange(-15.0, 15.0 + 1e-9, step))
    gz = 0.02 * gx ** 2 / 4 + 0.05 * gy
    for cx, cy, h, r in [(-12, 4, 3.0, 2.2), (-5, -6, 2.0, 1.8), (2, 7, 4.0, 2.6), (9, -2, 1.5, 1.2),
                         (15, 6, 2.5, 2.0), (-16, -9, 1.0, 1.5), (6, -10, 3.5, 3.0)]:
        gz += h * np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * r * r))
    verts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    ny, nx = gx.shape
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    faces = np.concatenate([np.stack([a, b, d], -1).reshape(-1, 3), np.stack([a, d, c], -1).reshape(-1, 3)])
    return verts, faces


def _write_stl(path, verts, faces):
    tri = verts[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    rec = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = nrm, tri
    with open(path, "wb") as f:
        f.write(b"smoke test".ljust(80, b" "))
        f.write(struct.pack("<I", len(faces)))
        f.write(rec.tobytes())


def _rigid(deg, axis, shift):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    t = math.radians(deg)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    m = np.eye(4)
    m[:3, :3] = np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * k @ k
    m[:3, 3] = shift
    return m


def t_register():
    """register (the AI align's registration, MeshRegAuto's command line): a copy of a
    bumpy surface moved by a known rigid move is brought back (mean error < 0.05 mm),
    in full mode and in ICP mode from a small offset; bad arguments: exit code 2, a
    missing file: 1. In the Cyrillic folder."""
    verts, faces = _bumpy_surface()
    target = os.path.join(folder, "дуга.stl")
    _write_stl(target, verts, faces)
    out = os.path.join(folder, "матрица.txt")

    def error(move, matrix):            # how far the matrix leaves the moved copy from home
        h = np.c_[verts, np.ones(len(verts))]
        back = (h @ move.T) @ matrix.T
        return np.linalg.norm(back[:, :3] - verts, axis=1)

    for mode, move, extra in (("full", _rigid(12.0, (0.3, 0.2, 1.0), (3.0, -2.0, 1.5)), []),
                              ("icp", _rigid(2.0, (1.0, 0.0, 0.2), (0.4, 0.3, -0.2)), ["--registration_mode", "icp"])):
        source = os.path.join(folder, f"скан_{mode}.stl")
        _write_stl(source, (np.c_[verts, np.ones(len(verts))] @ move.T)[:, :3], faces)
        if os.path.exists(out):
            os.remove(out)
        t = time.perf_counter()
        p = subprocess.run([exe, "register", source, target, "--transform_output", out, "--seed", "1"] + extra,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
        sec = time.perf_counter() - t
        if p.returncode != 0 or not os.path.exists(out):
            check(f"register {mode}", False, f"exit {p.returncode}: {p.stdout.decode(errors='replace')[-300:]}")
            continue
        e = error(move, np.loadtxt(out))
        check(f"register {mode} mode: the known move recovered", e.mean() < 0.05,
              f"mean {e.mean():.4f} mm, max {e.max():.4f} mm ({sec:.2f} s)")
    p = subprocess.run([exe, "register", target], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    check("register: bad arguments, exit code 2", p.returncode == 2, f"exit {p.returncode}")
    p = subprocess.run([exe, "register", os.path.join(folder, "нет.stl"), target], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    check("register: a missing file, exit code 1", p.returncode == 1, f"exit {p.returncode}")


def t_distance():
    # the unit square z = 0 (two triangles); points above, beside and past max_distance
    blocks = []

    def put(values, dtype):
        a = np.ascontiguousarray(values, dtype=dtype).reshape(-1, 3)
        info = eng.call("buffer.new", dtype=dtype, rows=int(a.shape[0]), cols=3)
        blocks.append(info)
        eng.array(info)[0][...] = a
        return info["volume"]

    co = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    faces = [[0, 1, 2], [0, 2, 3]]
    pts = [[0.5, 0.5, 0.3], [2.0, 0.5, 0.0], [1.6, 1.8, 0.0], [0.2, 0.7, -5.0]]
    want = [0.3, 1.0, 1.0, np.inf]
    res = eng.call("mesh.distance", points=put(pts, "float64"), vertices=put(co, "float32"),
                   faces=put(faces, "int32"), max_distance=2.5)
    d = eng.array(res["distances"]).reshape(-1).copy()
    blocks.append(res["distances"])
    ok = all((np.isinf(w) and np.isinf(g)) or abs(g - w) < 1e-5 for g, w in zip(d, want))
    check("mesh.distance (point to triangle, past max_distance: inf)", ok,
          f"{', '.join(f'{x:.4f}' for x in d)} (want 0.3, 1.0, 1.0, inf)")
    for info in blocks:
        eng.free(info)


run("engine starts", t_ping)
run("CT in and out", t_load)
run("loop cut", t_cut)
run("distance", t_distance)
run("thumbnails", t_thumbnails)
run("crop", t_crop)
run("register", t_register)
for d in sys.argv[1:]:
    run(f"DICOM {d}", lambda d=d: t_dicom(os.path.abspath(d)))

eng.stop()
shutil.rmtree(tmp, ignore_errors=True)
print(f"\n{'all checks passed' if not failed else 'FAILED: ' + ', '.join(failed)}")
sys.exit(1 if failed else 0)
