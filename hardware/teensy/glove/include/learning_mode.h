#ifndef LEARNING_MODE_H
#define LEARNING_MODE_H

#include <Arduino.h>

struct FingerInstructionSet {
    int FingerNumber; // 0 is thumb, 4 is pinky to match the select line mapping in firmware
    float DistanceToNote;
    SensorValue pressed; // after finger number is pressed, set this as either true or false and then sensor 
    // to finger number 

};

struct OctaveInstructionSet {
    float handPosition[2]; // 0 index is the x val, 
    // 1 index is the y val of the coord of the centroid of hand position for played note
};

#endif