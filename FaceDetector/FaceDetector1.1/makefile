#
# Make file for Face Detector
#
# Author: LI XUANPENG <marshallixp@gmail.com>
# Date: 20/9/2013

OCVFLAGS 	= `pkg-config --cflags opencv`
OCVLIBS 	= `pkg-config --libs opencv` -lstdc++

CPP 		= g++

# type of system?
LBITS 		= $(shell getconf LONG_BIT)
ifeq ($(LBITS),64)
# do 64 bit stuff here, like set some CFLAGS
	SFX = 64
else
	SFX = 32
endif

SRCS		 += DetectFace.cpp FaceDetectorTools.cpp GenerateFaceIndicators.cpp TrackFace.cpp Process_FaceDetectorSim.cpp main_FaceDetectorSim.cpp


OBJECTS 	+=  DetectFace.o FaceDetectorTools.o GenerateFaceIndicators.o  TrackFace.o Process_FaceDetectorSim.o main_FaceDetectorSim.o

CFLAGS		+= $(OCVFLAGS)
LIBS		+= $(OCVLIBS)
BINARY		= FaceDetectorSim$(SFX)

all: release

release: $(OBJECTS)
		$(CPP) $^ $(LDFLAGS) $(LIBS) $(CFLAGS) -o $(BINARY)

.cpp.o:
		$(CPP) $< $(CFLAGS) $(LIBS) $(LDFLAGS) -c -o $@

clean: 
		rm -f *.o $(OBJECTS) $(BINARY)
