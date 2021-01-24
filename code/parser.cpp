#include "utilities.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
using namespace std;

struct edge{
    int u, v, w;
    bool operator < (const edge &t) const{
        if (u != t.u) {
            return u < t.u;
        }else if (v != t.v) {
            return v < t.v;
        }
        return w < t.w;
    }
};

void writeVec(vector<int> &vec, string filename) {
    ofstream output(filename.c_str());
    int size = vec.size();
    for (int i = 0; i < size; ++i) {
        if (i) {
            output << ", ";
        }

        output << vec[i];
    }
    output << endl;
    output.close();
    cout << filename << " writing completed." << endl;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        cout << "Usage: ./parser FILE PATH\n"
                "FILE: file to be parsed\n"
                "PATH: directory to store the parsed result.\n"
                << endl;
        return -1;
    }

    string filename = argv[1];
    ifstream input(filename.c_str());
    if (!input.good()) {
        cout << "file does not exist!" << endl;
        return -1;
    }
    
    string outdir = argv[2];
    if (outdir.back() != '/') {
        outdir.push_back('/');
    }    

    char c;
    int u, v, w;
    string buf;
    vector<edge> vec; 

    while (getline(input, buf)) {
        if (buf[0] != 'a') {
            continue;
        }
        istringstream istr(buf);
        istr >> c >> u >> v >> w;
        vec.push_back({u, v, w});
    }
    input.close();

    stable_sort(vec.begin(), vec.end());

    vector<int> V, I, E, W, From, To;
    int pre = -1;
    int M = vec.size(); 
    for (int i = 0; i < M; ++i) {
        if (vec[i].u != pre) {
            pre = vec[i].u;
            V.push_back(vec[i].u);
            I.push_back(i);
        }
        E.push_back(vec[i].v);
        W.push_back(vec[i].w);
        From.push_back(vec[i].u);
        To.push_back(vec[i].v);
    }
    I.push_back(M);

    int N = V.size();
    cout << "vertex: " << N << " edges: " << M << endl;

    makeOutputFileName(filename);
    writeVec(V, outdir + filename + "_V.csv");
    writeVec(I, outdir + filename + "_I.csv");
    writeVec(E, outdir + filename + "_E.csv");
    writeVec(W, outdir + filename + "_W.csv");
    writeVec(From, outdir + filename + "_FROM.csv");
    writeVec(To, outdir + filename + "_TO.csv");

    return 0;
}