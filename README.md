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

A change of the protocol (a new command, other arguments) is a new engine version, and the
add-on that calls it raises its `engine.json` in the same pull request. `ODENT_ENGINE_API`
(main.cpp; `API` in the client) changes only when an older add-on cannot talk to it at all.

## Build on Linux

ITK, VTK and ONNX Runtime, once (about 22 min on 8 cores):

```
deps/build_deps.sh ~/MyApps/odent_engine_deps
```

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
