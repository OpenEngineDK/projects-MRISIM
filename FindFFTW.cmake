# - Try to find fftw
# Once done this will define
#
#  FFTW_FOUND - system has bullet
#  FFTW_INCLUDE_DIR - the bullet include directory
#  FFTW_LIBRARIES - Link these to use bullet
#

FIND_PATH(FFTW_INCLUDE_DIR NAMES fftw3
  PATHS
  ${FFTW_DEPS_INCLUDE_DIR}
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/libraries/fftw3/include
  ENV CPATH
  /usr/include
  /usr/local/include
  /opt/local/include
  NO_DEFAULT_PATH
)

FIND_LIBRARY(FFTW_LIBRARIES
  NAMES 
  fftw3
  PATHS
  ${FFTW_DEPS_LIB_DIR}
  ${PROJECT_BINARY_DIR}/lib
  ${PROJECT_SOURCE_DIR}/lib
  ${PROJECT_SOURCE_DIR}/libraries
  ${PROJECT_SOURCE_DIR}/libraries/fftw3/lib
  ENV LD_LIBRARY_PATH
  ENV LIBRARY_PATH
  /usr/lib
  /usr/local/lib
  /opt/local/lib
  NO_DEFAULT_PATH
)

SET(FFTW_FOUND FALSE)

IF(
    FFTW_INCLUDE_DIR AND
    FFTW_LIBRARIES
    )
SET(FFTW_FOUND TRUE)

ENDIF(
    FFTW_INCLUDE_DIR AND
    FFTW_LIBRARIES
    )

# show the BULLET_INCLUDE_DIR and BULLET_LIBRARIES variables only in the advanced view
IF(FFTW_FOUND)
  MARK_AS_ADVANCED(FFTW_INCLUDE_DIR FFTW_LIBRARIES )
ENDIF(FFTW_FOUND)
