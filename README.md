# ODent engine

The C++ program that does the CT and mesh work of the ODent5 Blender add-on: reading DICOM and
3D images, the slices, Dicom to Mesh, the Smart Cutter's cut, the Load Data previews, the
surface distances (heat maps), the AI teeth segmentation (`segment-teeth`) and the mesh
registration of the aligns (`register`). C++17, ITK 5.4.5 and VTK 9.7.1 (static), GDCM, ONNX
Runtime 1.23.2; GPL-3.0-or-later (LICENSE).

The add-on talks to it in JSON lines on its standard input and output, the voxels and meshes in
shared memory; `client/engine.py` is a copy of the add-on's client (the add-on's own is the one
in use). `segment-teeth` and `register` also run as commands of their own.

## Releases

Each version is released once, built and tested by GitHub Actions
(`.github/workflows/build.yml`), and that exact engine is what the add-on ships:

1. Raise `VERSION` (MAJOR.MINOR.PATCH), commit, push.
2. `git tag v<VERSION> && git push origin v<VERSION>`.
3. The workflow builds Linux and Windows, runs the smoke test, signs the Windows exe (once
   SignPath is set up) and publishes the release `v<VERSION>`: `engine-linux-x64.zip`,
   `engine-win-x64.zip` (`bin/<platform>/`) and a JSON text with their SHA-256.
4. In the ODent5 add-on, `engine.json` names the release to use: its release zips take that
   engine.

Then a pull request opens in the add-on's repository with `engine.json` pinned to it (the
workflow's `pin` job): try it, add the add-on's side of the change to that branch when there is
one, merge. Merging is what ships the engine to the team and the next ODent5 release.

### Versions: MAJOR.MINOR.PATCH

- **PATCH** (1.0.0 -> 1.0.1): a fix or a speed-up; the same commands and arguments. Any add-on
  that works with 1.0.0 works with it.
- **MINOR** (1.0.x -> 1.1.0): something new: a command, an argument, a field in a reply. Older
  add-ons still work; the add-on that uses the new part needs this version (its `engine.json`).
- **MAJOR** (1.x -> 2.0.0): the protocol breaks (a command removed or changed): raise
  `ODENT_ENGINE_API` too (main.cpp; `API` in the client), and older add-ons refuse the engine.

One version, one tag, one release: a released tag is never moved; a fix is the next PATCH.

## Build on Linux

ITK, VTK and ONNX Runtime, once (about 22 min on 8 cores):

```
deps/build_deps.sh ~/MyApps/odent_engine_deps
```

ONNX Runtime 1.23.2's original download (github.com/microsoft/onnxruntime) is gone since
2026-11-18. The same files, with the checksums the scripts pin, are this repository's release
`deps`: put the one for your platform in `<deps>/downloads/` first (the scripts then use it):
`gh release download deps -R OdentStudio/OdentEngine -p 'onnxruntime-linux-x64-*'
-D ~/MyApps/odent_engine_deps/downloads` (Windows: `-p 'onnxruntime-win-x64-*' -D
C:\odent_deps\downloads`). The workflow does the same.

Then the engine, installed into `bin/linux-x64/`, and the smoke test (a Python 3.13 with numpy:
Blender 5.2's Python works):

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$HOME/MyApps/odent_engine_deps/install
cmake --build build -j$(nproc)
cmake --install build --strip
python tests/smoke_test.py [DICOM folders]
```

The binary links libstdc++ in and needs glibc 2.38+. To try it in Blender, copy
`bin/linux-x64/*` into `addon/bin/linux-x64/` of an ODent5 checkout, or configure with
`-DODENT_INSTALL_DIR=<that folder>`; `ODENT_ENGINE_CLIENT=<checkout>/addon/core/engine.py`
runs the smoke test through the add-on's own client and bin/.

## Build on Windows

`build_windows.bat` does everything on a computer with Visual Studio (C++): BUILD_WINDOWS.md.

## Code signing policy

The Windows engine (`odent_engine.exe`) is signed. Free code signing is provided by
[SignPath.io](https://about.signpath.io), certificate by
[SignPath Foundation](https://signpath.org).

- Only files built by this repository's GitHub Actions workflow (`.github/workflows/build.yml`)
  from its own source, for a version tag, are signed, and every signing request is approved
  by hand.
- **Committers and reviewers:** the members of the
  [OdentStudio](https://github.com/OdentStudio) organisation with write access.
  Changes from anyone else are reviewed by one of them before they are merged.
- **Approvers:** Dr. Essaid Issam Dakir.

Third-party files shipped beside the engine keep their makers' signatures
(`onnxruntime.dll`: Microsoft).

## Privacy

The engine collects no data and makes **no network connection**. It reads and writes only the
files and memory the ODent5 add-on gives it, on your computer.
