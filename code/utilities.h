#ifndef BELLMAN_FORD_UTILITIES_H
#define BELLMAN_FORD_UTILITIES_H

#include <vector>
#include <fstream>
#include <iostream>
#include <string>

void loadVector(const char *filename, std::vector<int> &vec);
void printVector(std::vector<int> &vec);
void storeResult(const char *filename, std::vector<int> &V, int *D, int *P);
bool isValidFile(std::string filename);
void makeOutputFileName(std::string &inputFile);

#endif // BELLMAN_FORD_UTILITIES_H
