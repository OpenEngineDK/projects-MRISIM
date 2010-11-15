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
  Display/OpenGL/PhantomCanvas.h
  Display/OpenGL/PhantomCanvas.cpp

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
  Extensions_AntTweakBar
  Extensions_Inspection
  Extensions_InspectionBar
  Extensions_InspectionWidget
  Extensions_PropertyTree
  minc2 # ugly hack... use a find script
)
