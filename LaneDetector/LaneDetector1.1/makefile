#
# Make file for Lane Detector
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

SRCS		 += DetectLanes.cpp ExtractFeatures.cpp FittingCurve.cpp GenerateLaneIndicators.cpp IPMTransformation.cpp LaneDetectorTools.cpp TrackLanes.cpp main_LaneDetectorSim.cpp Process_LaneDetectorSim.cpp


OBJECTS 	+= DetectLanes.o ExtractFeatures.o FittingCurve.o GenerateLaneIndicators.o IPMTransformation.o LaneDetectorTools.o TrackLanes.o main_LaneDetectorSim.o Process_LaneDetectorSim.o

CFLAGS		+= $(OCVFLAGS)
LIBS		+= $(OCVLIBS)
BINARY		= LaneDetectorSim$(SFX)

all: release

release: $(OBJECTS)
		$(CPP) $^ $(LDFLAGS) $(LIBS) $(CFLAGS) -o $(BINARY)

.cpp.o:
		$(CPP) $< $(CFLAGS) $(LIBS) $(LDFLAGS) -c -o $@

clean: 
		rm -f *.o $(OBJECTS) $(BINARY)
