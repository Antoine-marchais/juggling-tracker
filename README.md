# Juggling Pattern Tracker

The goal of this project is to use computer vision to track and record the trajectory of juggling balls, in order to provide some insights to the juggler about the inconsistency in his juggling pattern.

This project relies heavily on the python implementation of opencv.

## Setup

We recommend using virtualenv to keep clean dependencies for each project.
To install all dependencies in your environment, simply run:

```
pip install -r requirements.txt
```

## Run

To run the tracker on a video, execute the following command:

```
python track.py  FILE_PATH [--start-frame INT] [--end-frame int] [--movement-path PATH]
[--keypoints-path PATH] [--trajectories-path PATH] [--save-fps INT] [--display-fps INT]
[--keypoints-method METHOD]
```

For a complete detail of the output of each step of the algorithm, we recommand specifying
`--movement-path` `--keypoints-path` and `--trajectories-path` to get the complete output.

For more information on how to use the arguments, run `python track.py --help`

## Results and Test Data

The __data__ folder contains four test videos:
- juggling_front.mp4
- juggling_side.mp4
- juggling_higher.mp4
- juggling_low_quality.mp4

These four videos allow to test the tracker on different juggling angles, heights, and video qualities.

Test results can be found in the __output__ folder.
