#include "utilities.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <climits>
#include <sstream>
#include <queue>
#include <algorithm>
using namespace std;

int n, m;
const int N = 1e7;
const int inf = 0x3f3f3f3f;
struct ac{
    int v, c;
    bool operator <(const ac &t) const{
        return t.c < c;
    }
};

vector<vector<ac>> g(N+1);
vector<int> dis(N+1, inf);
vector<bool> vis(N+1, false);
vector<int> pred(N+1, -1);

void Dijkstra() { // 堆优化  O(nlogn)
    dis[1] = 0;
    priority_queue<ac> que;
    que.push((ac){1, 0});
    while (!que.empty()) {
        ac f = que.top();
        que.pop();
        int u = f.v;
        if (dis[u] < f.c || vis[u]) continue;
        vis[u] = 1;
        for (int i = 0; i < (int)g[u].size(); ++i) {
            int v = g[u][i].v;
            int c = f.c + g[u][i].c;
            if (dis[v] > c) {
                dis[v] = c;
                que.push((ac){v, c});
                pred[v] = u;
            }else if (dis[v] == c) {
                pred[v] = min(pred[v], u);
            }
        }
    }
}

void intput(string &filename) {
    ifstream in(filename.c_str());
    if (!in.good()) {
        cout << "file does not exist!" << endl;
        exit(-1);
    }

    string buf, tmp;
    char c;
    int u, v, w;

    while (getline(in, buf)) {
        if (buf[0] == 'a') {
            istringstream istr(buf);
            istr >> c >> u >> v >> w;
            g[u].push_back({v, w});
        }else if (buf[0] == 'p') {
            istringstream istr(buf);
            istr >> c >> tmp >> n >> m;
        }
    }

    in.close();
}

void output(string &outdir, string &filename) {
    makeOutputFileName(filename);
    ofstream output(outdir + filename + "_dijkstra.csv");
    output << "Shortest Path: " << endl;
    for (int i = 1; i <= n; ++i) {
        // output << "from 1 to " << i << " = " << dis[i] << endl; 
        output << "from 1 to " << i << " = " << dis[i] << " predecessor = " << pred[i] << std::endl;
        
    }
    cout << outdir + filename + "_dijkstra.csv" << " writing completed." << endl;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        cout << "Usage: ./dijkstra FILE PATH\n"
                "FILE: input file\n"
                "PATH: directory to store the dijkstra result.\n"
                << endl;
        return -1;
    }

    string filename = argv[1];
    string outdir = argv[2];
    if (outdir.back() != '/') {
        outdir.push_back('/');
    }   

    intput(filename);

    cout << "nodes: " << n << " edges: " << m << endl;

    Dijkstra();
    cout << "dijkstra is done." << endl;
    
    output(outdir, filename);

    return 0;
}