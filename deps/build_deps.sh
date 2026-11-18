#!/usr/bin/env bash
# The ITK and VTK odent_engine is built with: pinned versions, static, only the
# modules it uses; and ONNX Runtime (prebuilt, a shared library shipped next to
# the engine) for the teeth segmentation (segment-teeth, the FastTeeth models). ITK 5.4 is what the addon's SimpleITK 2.5.6 is built on, VTK
# 9.7.1 is the addon's Python VTK (the loop cut needs 9.2+). About 22 min on 8
# cores; 1.1 GB of build files, 406 MB installed.
#
#   deps/build_deps.sh [<deps dir>]     (default ~/MyApps/odent_engine_deps)
#
# Then: cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_PREFIX_PATH=<deps dir>/install
set -euo pipefail

DEPS="${1:-$HOME/MyApps/odent_engine_deps}"
JOBS="${JOBS:-$(nproc)}"
VTK_VERSION=9.7.1
VTK_SHA256=cae04fd355004cb916a409db79d53a208f1221e975aeacc7540ee67b148ee91a
ITK_VERSION=5.4.5
ITK_SHA256=ecab9119664e2571b90740ba9ab3ca11cb46942dbd7bb87c0de5bb15309a36c9
# the version FastTeeth's Python build runs (its results are the reference)
ORT_VERSION=1.23.2
ORT_SHA256=1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b

mkdir -p "$DEPS"/{downloads,src,build,install}
fetch() {  # url file sha256
  if [ ! -f "$DEPS/downloads/$2" ]; then curl -fsSL -o "$DEPS/downloads/$2" "$1"; fi
  echo "$3  $DEPS/downloads/$2" | sha256sum -c -
}
fetch "https://vtk.org/files/release/${VTK_VERSION%.*}/VTK-$VTK_VERSION.tar.gz" "VTK-$VTK_VERSION.tar.gz" "$VTK_SHA256"
fetch "https://github.com/InsightSoftwareConsortium/ITK/releases/download/v$ITK_VERSION/InsightToolkit-$ITK_VERSION.tar.gz" \
      "InsightToolkit-$ITK_VERSION.tar.gz" "$ITK_SHA256"
[ -d "$DEPS/src/VTK-$VTK_VERSION" ] || tar -xzf "$DEPS/downloads/VTK-$VTK_VERSION.tar.gz" -C "$DEPS/src"
[ -d "$DEPS/src/InsightToolkit-$ITK_VERSION" ] || tar -xzf "$DEPS/downloads/InsightToolkit-$ITK_VERSION.tar.gz" -C "$DEPS/src"

COMMON=(-DCMAKE_BUILD_TYPE=Release "-DCMAKE_INSTALL_PREFIX=$DEPS/install"
        -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON)

cmake -S "$DEPS/src/VTK-$VTK_VERSION" -B "$DEPS/build/vtk" "${COMMON[@]}" \
  -DVTK_BUILD_ALL_MODULES=OFF -DVTK_BUILD_TESTING=OFF -DVTK_BUILD_EXAMPLES=OFF \
  -DVTK_ENABLE_WRAPPING=OFF -DVTK_WRAP_PYTHON=OFF -DVTK_WRAP_JAVA=OFF \
  -DVTK_GROUP_ENABLE_Imaging=DONT_WANT -DVTK_GROUP_ENABLE_MPI=DONT_WANT -DVTK_GROUP_ENABLE_Qt=DONT_WANT \
  -DVTK_GROUP_ENABLE_Rendering=DONT_WANT -DVTK_GROUP_ENABLE_StandAlone=DONT_WANT \
  -DVTK_GROUP_ENABLE_Views=DONT_WANT -DVTK_GROUP_ENABLE_Web=DONT_WANT \
  -DVTK_MODULE_ENABLE_VTK_CommonCore=YES -DVTK_MODULE_ENABLE_VTK_CommonDataModel=YES \
  -DVTK_MODULE_ENABLE_VTK_FiltersCore=YES -DVTK_MODULE_ENABLE_VTK_FiltersGeneral=YES \
  -DVTK_MODULE_ENABLE_VTK_FiltersModeling=YES -DVTK_MODULE_ENABLE_VTK_jsoncpp=YES \
  -DVTK_SMP_IMPLEMENTATION_TYPE=STDThread -DVTK_USE_CUDA=OFF -DVTK_USE_MPI=OFF
cmake --build "$DEPS/build/vtk" -j "$JOBS"
cmake --install "$DEPS/build/vtk"

cmake -S "$DEPS/src/InsightToolkit-$ITK_VERSION" -B "$DEPS/build/itk" "${COMMON[@]}" \
  -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DITK_BUILD_DEFAULT_MODULES=OFF -DITK_WRAP_PYTHON=OFF \
  -DITK_USE_SYSTEM_GDCM=OFF -DModule_ITKCommon=ON -DModule_ITKIOImageBase=ON -DModule_ITKIOGDCM=ON \
  -DModule_ITKIOMeta=ON -DModule_ITKIONRRD=ON -DModule_ITKIONIFTI=ON -DModule_ITKImageGrid=ON
cmake --build "$DEPS/build/itk" -j "$JOBS"
cmake --install "$DEPS/build/itk"
# ONNX Runtime: Microsoft's prebuilt CPU package (building it statically is a
# project of its own); CMake finds it in <install>/onnxruntime
fetch "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/onnxruntime-linux-x64-$ORT_VERSION.tgz" \
      "onnxruntime-linux-x64-$ORT_VERSION.tgz" "$ORT_SHA256"
rm -rf "$DEPS/install/onnxruntime"
tar -xzf "$DEPS/downloads/onnxruntime-linux-x64-$ORT_VERSION.tgz" -C "$DEPS/install"
mv "$DEPS/install/onnxruntime-linux-x64-$ORT_VERSION" "$DEPS/install/onnxruntime"
echo "ITK $ITK_VERSION, VTK $VTK_VERSION and ONNX Runtime $ORT_VERSION installed in $DEPS/install"
