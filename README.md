## Bellman-Ford

使用CUDA实现Bellman-Ford算法。



## Install

```bash
git clone git@github.com:Sanzona/Bellman-Ford.git

cd Bellman-Ford
```

![image-20210124173047451](img/image-20210124173047451.png)



## Compile

```bash
bash build.sh

cd build
```

![image-20210124173306208](img/image-20210124173306208.png)





## Run

```bash
# parse data
./parser ../data/USA-road-d.NY.gr input/

# run bellman
./bellman input/USA-road-d.NY.gr 512 1024 0

# dijkstra for test.
./dijkstra ../data/USA-road-d.NY.gr output
```



![image-20210124173538315](img/image-20210124173538315.png)



![image-20210124173820142](img/image-20210124173820142.png)



![image-20210124173416959](img/image-20210124173416959.png)



![image-20210124174029694](img/image-20210124174029694.png)



![image-20210124174610622](img/image-20210124174610622.png)



## References

[sengorajkumar/gpu_graph_algorithms](https://github.com/sengorajkumar/gpu_graph_algorithms)

[Bellman-Ford Single Source Shortest Path Algorithm on GPU using CUDA](https://towardsdatascience.com/bellman-ford-single-source-shortest-path-algorithm-on-gpu-using-cuda-a358da20144b)

