# Capstone Complete Coverage Path Planning

This repository contains code for complete coverage path planning for FLOOR-E, our oil tank floor inspection robot.

## Getting Started

### Installing and Dependencies

Install requirements.

```
pip install -r requirements.txt
```

You also need to have the yaml, and image files on the base directory. There are some files provided for testing purposes.

### Executing program

```
$ python3 image.py [image_name] [robot_width] [margin] [side_distance]
```

`image_name` should be the name of a yaml-image pair. The images can actually be any format supported by PIL, not necessarily pgm.

`robot_width` is the footprint of the robot in the dimension perpendicular to the direction of travel in pixels. It is assumed that the robot is capable of covering the entire area underneath it.

`margin` is the distance from obstacles.

`side_distance` is the distance to keep from obstacles on the side of the map.

Output is a list of lists of x,y-coordinate pairs for the robot to navigate between. The entire list describes a path that hopefully achieves complete coverage through the map.

The top level list contains sub-lists. Each sub-list describes a path throughout one boustrophedic cell.

The output is also saved to a json file `out_{mapname}.json` that can be passed to the onboard navigation software.

## Authors

[@Nikos](https://github.com/nik0sc)
[@Miguel](https://github.com/migsquizon)