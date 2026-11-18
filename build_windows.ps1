<#
odent_engine for Windows, in one go: downloads and builds the pinned ITK 5.4.5 and
VTK 9.7.1 (static, static C runtime, only the modules the engine uses), builds the
engine, installs it as bin\win-x64\odent_engine.exe and runs the smoke test.
The same versions and checksums as deps/build_deps.sh (Linux).

    build_windows.bat                         (double-click, or from cmd)
    powershell -ExecutionPolicy Bypass -File build_windows.ps1 [-Deps C:\odent_deps]
        [-Jobs 8] [-Python <python.exe>] [-Clean]

Needs: Visual Studio 2022 or newer (or its Build Tools) with "Desktop development
with C++"; CMake 3.20+ (the one inside Visual Studio is found). Windows 10 1903+.
The first run takes 30-60 min (ITK and VTK); later runs rebuild the engine only.
-Deps: keep it short (C:\odent_deps): ITK's build paths are long and Windows
stops at 260 characters. -Clean: build ITK and VTK again from scratch.
#>
param(
    [string]$Deps = "C:\odent_deps",
    [int]$Jobs = [Environment]::ProcessorCount,
    [string]$Python = "",
    [switch]$Clean
)
$ErrorActionPreference = "Stop"

$VTK_VERSION = "9.7.1"
$VTK_SHA256 = "cae04fd355004cb916a409db79d53a208f1221e975aeacc7540ee67b148ee91a"
$ITK_VERSION = "5.4.5"
$ITK_SHA256 = "ecab9119664e2571b90740ba9ab3ca11cb46942dbd7bb87c0de5bb15309a36c9"
# ONNX Runtime for segment-teeth (FastTeeth's models): the version build_deps.sh pins
$ORT_VERSION = "1.23.2"
$ORT_SHA256 = "0b38df9af21834e41e73d602d90db5cb06dbd1ca618948b8f1d66d607ac9f3cd"

$Engine = $PSScriptRoot

function Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function Fail($text) { Write-Host "`nERROR: $text" -ForegroundColor Red; exit 1 }
function Run {
    # a command, stopping the script when it fails
    & $args[0] $args[1..($args.Count - 1)]
    if ($LASTEXITCODE -ne 0) { Fail "failed ($LASTEXITCODE): $($args -join ' ')" }
}

# --- tools ------------------------------------------------------------------------
Step "Tools"
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { Fail "Visual Studio is not installed (vswhere.exe not found). Install Visual Studio 2022 Build Tools with 'Desktop development with C++'." }
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { Fail "Visual Studio has no C++ compiler: add 'Desktop development with C++' in the Visual Studio Installer." }
Write-Host "Visual Studio: $vs"

$cmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
if (-not $cmake) {
    $cmake = Join-Path $vs "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    if (-not (Test-Path $cmake)) { Fail "CMake not found: install it (cmake.org) or the 'C++ CMake tools' of Visual Studio." }
}
Write-Host "CMake: $cmake ($((& $cmake --version)[0]))"
foreach ($t in "tar.exe", "curl.exe") {
    if (-not (Get-Command $t -ErrorAction SilentlyContinue)) { Fail "$t not found (it comes with Windows 10 1803+)." }
}

# --- ITK and VTK -------------------------------------------------------------------
$install = Join-Path $Deps "install"
$stamp = Join-Path $install "odent_deps_itk${ITK_VERSION}_vtk${VTK_VERSION}.txt"
if ($Clean -and (Test-Path $Deps)) {
    Step "Removing $Deps\build and $Deps\install"
    Remove-Item -Recurse -Force (Join-Path $Deps "build"), $install -ErrorAction SilentlyContinue
}
if (Test-Path $stamp) {
    Step "ITK $ITK_VERSION and VTK $VTK_VERSION already built in $install"
} else {
    foreach ($d in "downloads", "src", "build", "install") { New-Item -ItemType Directory -Force (Join-Path $Deps $d) | Out-Null }

    function Fetch($url, $file, $sha) {
        $path = Join-Path $Deps "downloads\$file"
        if (-not (Test-Path $path)) {
            Write-Host "Downloading $file"
            Run curl.exe -fL --retry 3 -o $path $url
        }
        $got = (Get-FileHash -Algorithm SHA256 $path).Hash.ToLower()
        if ($got -ne $sha) { Remove-Item $path; Fail "${file}: checksum $got, expected $sha (download removed, run again)" }
        if (-not (Test-Path (Join-Path $Deps "src\$($file -replace '\.tar\.gz$','')"))) {
            Write-Host "Unpacking $file"
            Run tar.exe -xzf $path -C (Join-Path $Deps "src")
        }
    }
    Step "Sources"
    $vtkMinor = $VTK_VERSION.Substring(0, $VTK_VERSION.LastIndexOf("."))
    Fetch "https://vtk.org/files/release/$vtkMinor/VTK-$VTK_VERSION.tar.gz" "VTK-$VTK_VERSION.tar.gz" $VTK_SHA256
    Fetch "https://github.com/InsightSoftwareConsortium/ITK/releases/download/v$ITK_VERSION/InsightToolkit-$ITK_VERSION.tar.gz" "InsightToolkit-$ITK_VERSION.tar.gz" $ITK_SHA256

    # static libraries with the static C runtime (/MT), as the engine links them
    $common = @("-A", "x64", "-DCMAKE_INSTALL_PREFIX=$install", "-DBUILD_SHARED_LIBS=OFF",
                "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded", "-DCMAKE_POLICY_DEFAULT_CMP0091=NEW")

    Step "VTK $VTK_VERSION (about 10-20 min)"
    Run $cmake -S "$Deps\src\VTK-$VTK_VERSION" -B "$Deps\build\vtk" @common `
        -DVTK_BUILD_ALL_MODULES=OFF -DVTK_BUILD_TESTING=OFF -DVTK_BUILD_EXAMPLES=OFF `
        -DVTK_ENABLE_WRAPPING=OFF -DVTK_WRAP_PYTHON=OFF -DVTK_WRAP_JAVA=OFF `
        -DVTK_GROUP_ENABLE_Imaging=DONT_WANT -DVTK_GROUP_ENABLE_MPI=DONT_WANT -DVTK_GROUP_ENABLE_Qt=DONT_WANT `
        -DVTK_GROUP_ENABLE_Rendering=DONT_WANT -DVTK_GROUP_ENABLE_StandAlone=DONT_WANT `
        -DVTK_GROUP_ENABLE_Views=DONT_WANT -DVTK_GROUP_ENABLE_Web=DONT_WANT `
        -DVTK_MODULE_ENABLE_VTK_CommonCore=YES -DVTK_MODULE_ENABLE_VTK_CommonDataModel=YES `
        -DVTK_MODULE_ENABLE_VTK_FiltersCore=YES -DVTK_MODULE_ENABLE_VTK_FiltersGeneral=YES `
        -DVTK_MODULE_ENABLE_VTK_FiltersModeling=YES -DVTK_MODULE_ENABLE_VTK_jsoncpp=YES `
        -DVTK_SMP_IMPLEMENTATION_TYPE=STDThread -DVTK_USE_CUDA=OFF -DVTK_USE_MPI=OFF
    Run $cmake --build "$Deps\build\vtk" --config Release --parallel $Jobs
    Run $cmake --install "$Deps\build\vtk" --config Release

    Step "ITK $ITK_VERSION (about 20-40 min)"
    Run $cmake -S "$Deps\src\InsightToolkit-$ITK_VERSION" -B "$Deps\build\itk" @common `
        -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DITK_BUILD_DEFAULT_MODULES=OFF -DITK_WRAP_PYTHON=OFF `
        -DITK_USE_SYSTEM_GDCM=OFF -DModule_ITKCommon=ON -DModule_ITKIOImageBase=ON -DModule_ITKIOGDCM=ON `
        -DModule_ITKIOMeta=ON -DModule_ITKIONRRD=ON -DModule_ITKIONIFTI=ON -DModule_ITKImageGrid=ON `
        -DITK_MSVC_STATIC_RUNTIME_LIBRARY=ON
    Run $cmake --build "$Deps\build\itk" --config Release --parallel $Jobs
    Run $cmake --install "$Deps\build\itk" --config Release

    Set-Content -Path $stamp -Value "ITK $ITK_VERSION, VTK $VTK_VERSION, static, /MT, built $(Get-Date -Format s)"
}

# --- ONNX Runtime (prebuilt: onnxruntime.dll beside the engine) ---------------------
$ort = Join-Path $install "onnxruntime"
if (-not (Test-Path (Join-Path $ort "lib\onnxruntime.dll"))) {
    Step "ONNX Runtime $ORT_VERSION"
    $zip = Join-Path $Deps "downloads\onnxruntime-win-x64-$ORT_VERSION.zip"
    New-Item -ItemType Directory -Force (Join-Path $Deps "downloads") | Out-Null
    if (-not (Test-Path $zip)) {
        Run curl.exe -fL --retry 3 -o $zip "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/onnxruntime-win-x64-$ORT_VERSION.zip"
    }
    $got = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
    if ($got -ne $ORT_SHA256) { Remove-Item $zip; Fail "onnxruntime zip: checksum $got, expected $ORT_SHA256 (download removed, run again)" }
    Remove-Item -Recurse -Force $ort -ErrorAction SilentlyContinue
    Expand-Archive -Path $zip -DestinationPath $install -Force
    Rename-Item (Join-Path $install "onnxruntime-win-x64-$ORT_VERSION") "onnxruntime"
}

# --- the engine -----------------------------------------------------------------------
Step "odent_engine"
$build = Join-Path $Engine "build-win"
Run $cmake -S $Engine -B $build -A x64 "-DCMAKE_PREFIX_PATH=$install"
Run $cmake --build $build --config Release --parallel $Jobs
Run $cmake --install $build --config Release
$exe = Join-Path $Engine "bin\win-x64\odent_engine.exe"
if (-not (Test-Path $exe)) { Fail "the install step did not make $exe" }
$mb = [math]::Round((Get-Item $exe).Length / 1MB)
Write-Host "Installed: $exe ($mb MB)"
$ping = '{"id":1,"cmd":"ping"}' | & $exe
Write-Host "ping: $ping"
if ($ping -notmatch '"api"') { Fail "the engine does not answer" }

# --- smoke test -----------------------------------------------------------------------
Step "Smoke test"
if (-not $Python) {
    # Blender's own Python has numpy; else any Python with numpy
    $blender = Get-ChildItem -Path "C:\blender*\*\python\bin\python.exe", "$env:ProgramFiles\Blender Foundation\*\*\python\bin\python.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($blender) { $Python = $blender.FullName }
    elseif (Get-Command py -ErrorAction SilentlyContinue) { $Python = "py" }
    elseif (Get-Command python -ErrorAction SilentlyContinue) { $Python = "python" }
}
if (-not $Python) {
    Write-Host "No Python found: run the smoke test by hand with -Python <blender>\5.2\python\bin\python.exe" -ForegroundColor Yellow
} else {
    Write-Host "Python: $Python"
    & $Python (Join-Path $Engine "tests\smoke_test.py")
    if ($LASTEXITCODE -ne 0) { Fail "the smoke test failed (see above)" }
}
Write-Host "`nDone: $exe" -ForegroundColor Green
Write-Host "Release zip: python tools\make_release.py --platform win-x64"
