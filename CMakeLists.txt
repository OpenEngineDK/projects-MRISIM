INCLUDE(${QT_USE_FILE})
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR})
# Project name. Should be the same as the project directory name and
# will become the name of the binary target.
SET( PROJECT_NAME "MRISIM")

QT4_WRAP_UI(UI_SOURCE
  MRIUI.ui
)
QT4_WRAP_CPP(MOC_SOURCE
  MRIUI.h
)


# Project source code list
SET( PROJECT_SOURCES
  # Add all the cpp source files here
  # main.cpp
  Resources/MINCResource.h
  Resources/MINCResource.cpp

  Resources/Phantom.h
  Resources/Phantom.cpp
  Resources/IPhantomBuilder.h
  Resources/SimplePhantomBuilder.h
  Resources/SimplePhantomBuilder.cpp
  Resources/MINCPhantomBuilder.h
  Resources/MINCPhantomBuilder.cpp
  Resources/Sample3DTexture.h
  Resources/Sample3DTexture.cpp


  Display/CanvasSwitch.h
  Display/CanvasSwitch.cpp
  Display/OpenGL/PhantomCanvas.h
  Display/OpenGL/PhantomCanvas.cpp
  Display/OpenGL/SpinCanvas.h
  Display/OpenGL/SpinCanvas.cpp
  Display/OpenGL/WindowCanvas.h
  Display/OpenGL/WindowCanvas.cpp
  Science/Science.cpp
  Science/MRISim.h
  Science/MRISim.cpp
  Science/CPUKernel.h
  Science/CPUKernel.cpp
  Science/UniformSampler.h
  Science/UniformSampler.cpp
  Science/FrequencyData.h
  Science/FrequencyData.cpp
  Science/SpinEchoSequence.h
  Science/SpinEchoSequence.cpp
  Science/ListSequence.h
  Science/ListSequence.cpp
  Science/IRFCoil.h
  Science/TestRFCoil.h
  Science/TestRFCoil.cpp

  Science/RFTester.h
  Science/RFTester.cpp

  Science/ExcitationPulseSequence.h
  Science/ExcitationPulseSequence.cpp

  # Science/FFTData.h
  # Science/FFTData.cpp

  Science/IFFT.h
  Science/CPUFFT.h
  Science/CPUFFT.cpp
  Science/ImageFFT.h
  Science/ImageFFT.cpp
  Science/CartesianFFT.h
  Science/CartesianFFT.cpp

  Scene/SpinNode.h
  Scene/SpinNode.cpp

  Science/BlochTest.h
  Science/BlochTest.cpp

  Science/OpenCLTest.h
  Science/OpenCLTest.cpp

  ${MOC_SOURCE}
  ${UI_SOURCE}
  MRIUI.cpp
)

# Include needed to use SDL under Mac OS X
# IF(APPLE)
#   SET(PROJECT_SOURCES ${PROJECT_SOURCES}  ${SDL_MAIN_FOR_MAC})
# ENDIF(APPLE)

# Project executable
ADD_EXECUTABLE(${PROJECT_NAME}
  ${PROJECT_SOURCES}
)

# Project dependencies
TARGET_LINK_LIBRARIES(${PROJECT_NAME}
  # Core library dependencies
  OpenEngine_Core
  OpenEngine_Logging
  OpenEngine_Display
  OpenEngine_Devices
  OpenEngine_Scene
  # Extension dependencies
  Extensions_Qt
  Extensions_MeshUtils
  Extensions_SetupHelpers  
  Extensions_Plotting
  Extensions_MathGL
  Extensions_Wall
  Extensions_SDLFont
  Extensions_Inspection
  Extensions_InspectionWidget
  Extensions_PropertyTree
  Extensions_OpenCL
  minc2 # ugly hack... use a find script
  fftw3
)
