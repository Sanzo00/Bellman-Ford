#include "main.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        cout << "Usage: ./bellman FILE BLOCKS BLOCK_SIZE DEBUG" << endl;
        cout << "FILE: Input file\n"
                "BLOCKS: Number of blocks for cuda\n"
                "BLOCK_SIZE: Number of threads per block for cuda\n"
                "DEBUG: 1 or 0 to enable/disable extended debug messages on console\n"
                "Program expects these CSV files based on FILE thats passed in thre argument\n"
                "   FILE_V.csv\n"
                "   FILE_I.csv\n"
                "   FILE_E.csv\n"
                "   FILE_W.csv\n"
                << endl;
        return -1;
    }
    
    std::string filename = argv[1];

    if (!isValidFile(filename + "_V.csv") ||
        !isValidFile(filename + "_I.csv") ||
        !isValidFile(filename + "_E.csv") ||
        !isValidFile(filename + "_W.csv")) {

        cout << "One or more CSR files missing" << endl;
        return -1;
    }

    int BLOCKS = argc > 2 ? atoi(argv[2]) : 512;
    int BLOCK_SIZE = argc > 3 ? atoi(argv[3]) :1024;
    int debug = argc > 4 ? atoi(argv[4]) : 0;

    runBellmanFordOnGPU(filename.c_str(), BLOCKS, BLOCK_SIZE, debug);

    return 0;
}
