cmake_minimum_required(VERSION 3.20)
# C: Ubuntu's ITK loads its VTK bridge, whose MPI lookup needs it
project(odent_engine VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The protocol version the addon checks in its handshake (engine.py).
set(ODENT_ENGINE_API 1)

# ITK: DICOM (GDCM), MetaImage, NRRD, NIfTI, resampling.
find_package(ITK REQUIRED COMPONENTS
  ITKCommon ITKIOImageBase ITKIOGDCM ITKIOMeta ITKIONRRD ITKIONIFTI ITKImageGrid)
include(${ITK_USE_FILE})
# GDCM directly: its header scanner checks a series before it is read. ITK's
# own GDCM comes with ITK_LIBRARIES; Ubuntu's ITK uses the system one.
set(ODENT_GDCM "")
if(GDCM_DIR)   # set by an ITK built on the system GDCM (Ubuntu)
  find_package(GDCM REQUIRED)
  include(${GDCM_USE_FILE})
  set(ODENT_GDCM gdcmMSFF)
endif()
# VTK: meshes from the CT, the loop cut; jsoncpp for the protocol.
find_package(VTK REQUIRED COMPONENTS
  CommonCore CommonDataModel FiltersCore FiltersGeneral FiltersModeling jsoncpp)

add_executable(odent_engine
  src/main.cpp
  src/shm.cpp
  src/volume.cpp
  src/image_commands.cpp
  src/slice_commands.cpp
  src/mesh_commands.cpp
  src/mesh_io.cpp
  src/mesh_preview.cpp
  src/registration.cpp
  src/register_main.cpp)
# nanoflann (header only, BSD): the registration's KD-trees, as in Open3D 0.19
target_include_directories(odent_engine PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nanoflann")
target_compile_definitions(odent_engine PRIVATE ODENT_ENGINE_API=${ODENT_ENGINE_API})
# the engine's own version (VERSION, MAJOR.MINOR.PATCH): the release tag v<version>, the ping's
# "version", the Windows version information; the ODent5 add-on pins it (its engine.json)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" ODENT_ENGINE_VERSION LIMIT_COUNT 1)
string(STRIP "${ODENT_ENGINE_VERSION}" ODENT_ENGINE_VERSION)
if(NOT ODENT_ENGINE_VERSION MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$")
  message(FATAL_ERROR "VERSION: '${ODENT_ENGINE_VERSION}' is not MAJOR.MINOR.PATCH")
endif()
set(ODENT_V_MAJOR ${CMAKE_MATCH_1})
set(ODENT_V_MINOR ${CMAKE_MATCH_2})
set(ODENT_V_PATCH ${CMAKE_MATCH_3})
target_compile_definitions(odent_engine PRIVATE ODENT_ENGINE_VERSION="${ODENT_ENGINE_VERSION}")
if(WIN32)
  # UTF-8 file names (Cyrillic DICOM folders) and long paths: see the manifest
  target_sources(odent_engine PRIVATE src/odent_engine.manifest)
  # version information (product ODent engine, VERSION as its version): code signing wants
  # it the same in every build of a release
  configure_file(src/odent_engine.rc.in "${CMAKE_CURRENT_BINARY_DIR}/odent_engine.rc" @ONLY)
  target_sources(odent_engine PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/odent_engine.rc")
endif()
# The Smart Cutter's cut needs vtkSelectPolyData's Dijkstra edge search (VTK 9.2+):
# without it (Ubuntu's VTK 9.1) the engine is built without mesh.cut_loop.
if(VTK_VERSION VERSION_GREATER_EQUAL 9.2)
  target_sources(odent_engine PRIVATE src/cut_commands.cpp)
  target_compile_definitions(odent_engine PRIVATE ODENT_HAS_CUT=1)
else()
  message(WARNING "VTK ${VTK_VERSION}: the engine is built without the loop cut (needs 9.2+)")
endif()
target_link_libraries(odent_engine PRIVATE ${ITK_LIBRARIES} ${VTK_LIBRARIES} ${ODENT_GDCM})
if(UNIX AND NOT APPLE)
  target_link_libraries(odent_engine PRIVATE rt)
  # libstdc++ and libgcc inside the binary: it then needs only glibc (2.38 when
  # built on Ubuntu 24.04), not the GCC 13 runtime of the build machine
  target_link_options(odent_engine PRIVATE -static-libstdc++ -static-libgcc)
endif()
if(MSVC)
  # the C runtime inside the .exe (/MT): no Visual C++ redistributable to install.
  # The ITK and VTK it links must be built the same way (build_deps.bat does).
  set_property(TARGET odent_engine PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
  # Windows headers: no min/max macros (they break std::min), fewer includes
  target_compile_definitions(odent_engine PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  target_compile_options(odent_engine PRIVATE /W3 /utf-8 /bigobj)
endif()
# ONNX Runtime (prebuilt, engine/deps): the teeth segmentation (segment-teeth,
# FastTeeth's models). Shipped as a shared library next to the engine; without it
# the engine is built without segment-teeth.
find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h PATH_SUFFIXES onnxruntime/include include)
find_library(ONNXRUNTIME_LIB onnxruntime PATH_SUFFIXES onnxruntime/lib lib)
if(ONNXRUNTIME_INCLUDE AND ONNXRUNTIME_LIB)
  target_sources(odent_engine PRIVATE src/teeth_segment.cpp)
  target_include_directories(odent_engine PRIVATE ${ONNXRUNTIME_INCLUDE})
  target_link_libraries(odent_engine PRIVATE ${ONNXRUNTIME_LIB})
  target_compile_definitions(odent_engine PRIVATE ODENT_HAS_TEETH=1)
  if(UNIX)   # the library next to the binary
    set_target_properties(odent_engine PROPERTIES BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
  endif()
else()
  message(WARNING "ONNX Runtime not found: the engine is built without segment-teeth")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(odent_engine PRIVATE -Wall -Wextra)
endif()
vtk_module_autoinit(TARGETS odent_engine MODULES ${VTK_LIBRARIES})

# `cmake --install build` puts the binary in bin/<platform>/ (gitignored): the layout of the
# release zips and of the add-on's bin/. -DODENT_INSTALL_DIR=<add-on>/bin/<platform> installs it
# straight into an ODent5 add-on checkout to try it in Blender.
if(WIN32)
  set(ODENT_PLATFORM win-x64)
else()
  set(ODENT_PLATFORM linux-x64)
endif()
set(ODENT_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bin/${ODENT_PLATFORM}" CACHE PATH
    "where cmake --install puts the engine and ONNX Runtime")
get_filename_component(ODENT_ADDON_BIN "${ODENT_INSTALL_DIR}" ABSOLUTE)
install(TARGETS odent_engine RUNTIME DESTINATION "${ODENT_ADDON_BIN}")
if(ONNXRUNTIME_LIB)
  # the ONNX Runtime library the engine loads, beside it (Linux: the .so.1 the
  # binary names; Windows: onnxruntime.dll from the package's lib folder)
  get_filename_component(_ort_dir "${ONNXRUNTIME_LIB}" DIRECTORY)
  if(WIN32)
    install(FILES "${_ort_dir}/onnxruntime.dll" DESTINATION "${ODENT_ADDON_BIN}")
  else()
    file(GLOB _ort_so "${_ort_dir}/libonnxruntime.so.*.*.*")
    install(FILES ${_ort_so} DESTINATION "${ODENT_ADDON_BIN}" RENAME libonnxruntime.so.1)
  endif()
endif()
