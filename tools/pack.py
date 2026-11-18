"""The release zip of one platform: engine-<platform>.zip holding bin/<platform>/ (the engine,
ONNX Runtime and engine_source.txt = the release tag), the layout of the add-on's bin/.

    python tools/pack.py <platform> <tag>        (linux-x64 | win-x64, e.g. v1.0.0)

Prints the zip's SHA-256 and, in GitHub Actions, gives it as the step's output `sha256`.
The executable bit is kept (the add-on also restores it: Blender's zip install drops it).
"""

import hashlib
import os
import sys
import zipfile

platform, tag = sys.argv[1], sys.argv[2]
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder = os.path.join(root, "bin", platform)
with open(os.path.join(folder, "engine_source.txt"), "w") as f:
    f.write(tag)
name = f"engine-{platform}.zip"
with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as zf:
    for entry in sorted(os.listdir(folder)):
        src = os.path.join(folder, entry)
        info = zipfile.ZipInfo.from_file(src, f"bin/{platform}/{entry}")
        info.compress_type = zipfile.ZIP_DEFLATED
        with open(src, "rb") as fh:
            zf.writestr(info, fh.read())
digest = hashlib.sha256(open(name, "rb").read()).hexdigest()
if os.environ.get("GITHUB_OUTPUT"):
    with open(os.environ["GITHUB_OUTPUT"], "a") as out:
        out.write(f"sha256={digest}\n")
print(name, digest)
