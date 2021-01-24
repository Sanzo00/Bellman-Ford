#include "utilities.h"

void loadVector(const char *filename, std::vector<int> &vec) {
    std::ifstream input;
    input.open(filename);
    int num;
    while ((input >> num) && input.ignore()) {
        vec.push_back(num);
    }
    input.close();
}

void printVector(std::vector<int> &vec) {
    for (int i = 0; i < (int) vec.size(); ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

void storeResult(const char *filename, std::vector<int> &V, int *D, int *P) {
    std::ofstream output(filename);
    output << "Shortest Path: " << std::endl;
    for (int i = 0; i < V.size(); ++i) {
        // output << "from " << V[0] << " to " << V[i] << " = " << D[i] << std::endl;
        output << "from " << V[0] << " to " << V[i] << " = " << D[i] << " predecessor = " << P[i] << std::endl;
    }
    output.close();
}

bool isValidFile(std::string filename) {
    std::ifstream file(filename);
    return file.good();
}

void makeOutputFileName(std::string &inputFile){
    std::string delimiter = "/";
    size_t pos = inputFile.rfind(delimiter);
    if (pos != std::string::npos) {
        inputFile.erase(0, pos + delimiter.length());
    }
}
