# hyperbolic_alignment
This repo contains a base implementation for [aligning hyperbolic representations using an Optimal Transport-based approach](http://arxiv.org/abs/2012.01089).


# Examples: Hyperbolic/Euclidean Mapping Estimation 

This code is written in Python and relies on Anaconda to create an environment to install dependencies.

## Installing the anaconda environment.

All associated packages are opensource. The file environment.yml contains all the packages used in the examples. 

To install these packages run
```
make install
```

## Installing the anaconda environment.

It contains two examples, one for each use case. 

### Barycenter of shapes: interpolation of clounds of points

```
make run_example
```

----
## Linear and wrapped linear mapping
![alt text](linear_estimation.png)

## OT-DA
![alt text](nearest_neighbors_estimation.png)

## Hyp-ME
![alt text](HNN_mapping_estimation.png)


### Gyrobarycenter with various costs
![alt text](gyrobarycenter_map_with_different_costs.png)

----
This code is going to be shared in a Github repository after receiving the conference decision.