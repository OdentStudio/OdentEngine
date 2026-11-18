# Building odent_engine on Windows

The ODent5 add-on does the CT and mesh work (reading DICOM, the slices, Dicom to Mesh, the
Smart Cutter's cut, the AI teeth, the aligns) in `odent_engine.exe`. Releases are built by
GitHub Actions (README.md); this script builds it on a Windows computer, to work on the
engine. The Linux engine is built with `deps/build_deps.sh` and CMake (README.md).

## 1. Install the tools (once)

- **Visual Studio 2022 or newer**, or the free **Build Tools for Visual Studio**
  (visualstudio.microsoft.com/downloads, under "Tools for Visual Studio"). In the
  installer choose **Desktop development with C++**. That brings the compiler, the
  Windows SDK and CMake.
- **Git** to get the repository (git-scm.com), or download it as a zip from GitHub.
- Windows 10 version 1903 or newer (the engine reads file names as UTF-8: Cyrillic
  folder names work).
- About 6 GB of free disk space on `C:` for the build files of ITK and VTK.

Python is not needed: the script finds Blender's own Python for the test
(`C:\blender-*\5.2\python\bin\python.exe` or the installed Blender).

## 2. Build

```
git clone https://github.com/OdentStudio/OdentEngine.git C:\OdentEngine
C:\OdentEngine\build_windows.bat
```

(Double-clicking `build_windows.bat` in Explorer does the same.)

The script:

1. checks Visual Studio and CMake;
2. downloads ITK 5.4.5 and VTK 9.7.1 into `C:\odent_deps`, checks their SHA-256 and
   builds them: static libraries, static C runtime, only the modules the engine
   uses (the same versions and options as the Linux build). **30-60 minutes the
   first time**, then skipped;
3. builds the engine and installs it as `bin\win-x64\odent_engine.exe` with
   `onnxruntime.dll` beside it (no Visual C++ redistributable needed);
4. runs `tests\smoke_test.py`: starts the engine and checks the shared
   memory both ways, a Cyrillic folder, writing and reading a CT, marching cubes,
   the slices and the loop cut. Every line must say `OK`.

Options (after the `.bat` or with `powershell -ExecutionPolicy Bypass -File
build_windows.ps1`): `-Deps D:\odent_deps` (keep the path short: Windows
stops at 260 characters and ITK's build paths are long), `-Jobs 4` (fewer
processor cores, less memory), `-Python <python.exe>`, `-Clean` (build ITK and VTK
again).

To test on real DICOM data too:

```
C:\blender-5.2.2\5.2\python\bin\python.exe tests\smoke_test.py "D:\cases\<case>\DICOM"
```

## 3. Try it in Blender

Copy `bin\win-x64\*` into `addon\bin\win-x64\` of an ODent5 add-on checkout linked
into Blender. Start Blender and, on a saved test project: Load Data with a CT, move
the slices pointer, Dicom to Mesh, cut a scan with the Smart Cutter. Open Log shows any engine
error. A release build is signed by the workflow (README.md); a local one is not.

## When something fails

- **"Visual Studio has no C++ compiler"**: open the Visual Studio Installer, Modify,
  tick "Desktop development with C++".
- **Checksum error**: the download was cut; the script removed it, run it again.
- **A path too long / "cannot open file" deep in ITK**: use a shorter `-Deps`.
- **LNK2038 "mismatch detected for RuntimeLibrary"**: ITK or VTK were built with
  another runtime; run with `-Clean`.
- **The smoke test fails on the Cyrillic folder**: Windows older than 10 version
  1903 (the UTF-8 manifest needs it).
- The ITK and VTK builds write their logs in `C:\odent_deps\build\itk` and `\vtk`;
  the engine's in `build-win`.
