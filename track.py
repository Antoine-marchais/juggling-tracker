# -*- coding: utf-8 -*-
import cv2
import numpy as np
import utils
from argparse import ArgumentParser

def get_foreground_masks(frames):
    """

    :param frames:
    :return:
    """
    backsub = cv2.createBackgroundSubtractorMOG2(history=3, varThreshold=10)
    # training the substractor on the 30 middle frames
    start = max(int(len(frames)/2 - 15), 0)
    end = min(int(len(frames)/2 + 15), len(frames))
    for frame in frames[start:end]:
        backsub.apply(frame)

    # getting foreground masks
    foregrounds = []
    for frame in frames:
        foregrounds.append(backsub.apply(frame))
    return foregrounds


def blobify(img):
    """

    :param img:
    :return:
    """
    kernel_erode = np.ones((3,3), dtype=np.uint8)
    kernel_dilate = np.ones((3,3), dtype=np.uint8)
    res = img.copy()
    for _ in range(4):
        res= cv2.erode(res, kernel_erode, iterations=2)
        res = cv2.dilate(res, kernel_dilate, iterations=2)
    return res


def get_keypoints_hough(img):
    """get keypoints using the Hough transfrom for circle detection

    :param img: np.ndarray, image to get keypoints from
    :return: list(list(float, float)), list of keypoints in the image
    """
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=4, minDist=100, param1=100, param2=80, minRadius=10, maxRadius=50)
    keypoints = [[x, y] for (x, y, r) in circles[0, :]] if circles is not None else []
    return keypoints


def get_blob_detector():
    """

    :return:
    """
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255

    params.filterByCircularity = True
    params.minCircularity = 0.3

    params.filterByArea = True
    params.minArea = 1000
    params.maxArea = 1000000

    params.filterByConvexity = False

    params.filterByInertia = False
    params.minInertiaRatio = 0.2

    params.minDistBetweenBlobs = 100

    detector = cv2.SimpleBlobDetector_create(params)
    return detector


def draw_keypoints(img, keypoints, color=None, radius=None):
    """

    :param img:
    :param keypoints:
    :param color:
    :param radius:
    :return:
    """
    if color == None:
        color=np.random.randint(0, 256, size=3).tolist()
    if radius == None:
        radius = img.shape[0]//70
    result = np.copy(img)
    for keypoint in keypoints:
        cv2.circle(result, (int(keypoint[0]), int(keypoint[1])), radius, color=color, thickness=-1)
    return result


class Trajectory:
    """
    """
    def __init__(self, frame_i, start_point):
        self.points_dict = {frame_i: start_point}
        self.staleness = 0
    
    def add_point(self, frame, point):
        self.points_dict[frame] = point
        
    def is_active(self, frame, max_staleness):
        if frame < self.span[0]:
            return False
        if self.staleness >= max_staleness:
            return False
        return True
    
    
    def next_found_frame(self, frame_idx):
        for frame in self.found_frames:
            if frame > frame_idx:
                return frame
        return None
    
    @property
    def found_frames(self):
        return sorted(self.points_dict.keys())
    
    @property
    def span(self):
        found_frames = self.found_frames
        return list(range(found_frames[0], found_frames[-1] + 1))
    
    @property
    def points(self):
        return self.points_dict.items()
    
    @property
    def start_frame(self):
        return self.span[0]


def filter_valid_keypoints(previous_coordinates, keypoints, x_increasing, y_increasing):
    """Determine which of a frame’s keypoints are valid given the balls trajectory directions.
    
    Uses the previous ball coordinates and the trajectory directions to filter out keypoints
    that can’t be valid.
    :param list previous_coordinates: coordinates of the balls last keypoints.
    :param list keypoints: all the keypoint coordinates in the given frame.
    :param bool x_increasing: if True, the trajectory’s x coordinates are increasing.
    :param bool y_increasing: if True, the trajectory’s y coordinates are increasing.
    """
    remove_list = []
    direction_leeway = 10  # Direction leeway in pixels
    for i, keypoint in enumerate(keypoints):
        if x_increasing is not None:  
            # If x coordinates are increasing, filter keypoints with lower x     
            if x_increasing and keypoint[0] < previous_coordinates[0] - direction_leeway:
                remove_list.append(i)
            # If x coordinates are decreasing, filter keypoints with higher x
            elif not x_increasing and keypoint[0] > previous_coordinates[0] + direction_leeway:
                remove_list.append(i)
        if y_increasing is not None:
            # If y coordinates are increasing (ball falling), filter keypoints with lower y
            # Cannot filter keypoints if y is decreasing as the ball with change direction at some point
            if y_increasing and keypoint[1] < previous_coordinates[1]:
                remove_list.append(i)
    return [coordinates for i, coordinates in enumerate(keypoints) if i not in remove_list]


def closest_ball_index(previous_coordinates, valid_keypoints, max_dist):
    """

    :param previous_coordinates:
    :param valid_keypoints:
    :param max_dist:
    :return:
    """
    if not valid_keypoints:
        return None
    diff = np.array(valid_keypoints)-np.array(previous_coordinates)
    distances = [np.linalg.norm(point_diff) for point_diff in diff]
    if [dist for dist in distances if dist < max_dist]:
        return distances.index(min(distances))
    return None


def identify_ball(previous_coordinates, valid_keypoints, max_dist=200):
    """Identify the next keypoint in trajectory.
    
    Returns the closest valid keypoint.
    :param list previous_coordinates: coordinates of the balls last keypoints.
    :param list valid_keypoints: all the valid keypoint coordinates in the given frame.
    :param int max_dist: maximum distance of valid keypoint from previous keypoint in pixels.
    """
    closest = closest_ball_index(previous_coordinates, valid_keypoints, max_dist)
    if closest is not None:
        return valid_keypoints[closest]
    return None


def track_single_ball(all_keypoints, frame_idx, keypoint_idx_in_first_frame, max_lost_frame_nb=5,
                     remove_keypoints=False):
    """Tracks a single ball’s keypoint trajectory.
    
    :param list all_keypoints: all remaining keypoints.
    :param int frame_idx: index of frame with the first keypoint.
    :param int keypoint_idx_in_first_frame: index of the keypoint in first frame.
    :param int max_lost_frame_nb: maximum number of frames to continue searching for ball trajectory.
    :param bool remove_keypoints: if True, keypoints that are used in the trajectory are removed from
    the keypoints list.
    """ 
    previous_coordinates = all_keypoints[frame_idx][keypoint_idx_in_first_frame]
    trajectory = Trajectory(frame_idx, previous_coordinates)
    if remove_keypoints:
        all_keypoints[frame_idx].remove(previous_coordinates)
    nb_frames_ball_lost = 0
    # Keep track of trajectory direction
    x_increasing = None
    y_increasing = None
    # Keypoints belonging to the ball’s trajectory
    while nb_frames_ball_lost < max_lost_frame_nb and frame_idx < len(all_keypoints) - 1:
        next_keypoints = all_keypoints[frame_idx + 1]  # All keypoints in the next frame
        # Determine which keypoints are valid given the balls trajectory directions
        valid_keypoints = filter_valid_keypoints(previous_coordinates, next_keypoints,
                                                 x_increasing, y_increasing)
        # Determine the next keypoint in the balls trajectory
        new_coordinates = identify_ball(previous_coordinates, valid_keypoints)
        if new_coordinates is not None:
            # Update trajectory directions and keypoint variables
            if x_increasing is None:
                x_increasing = True if new_coordinates[0] > previous_coordinates[0] else False
            y_increasing = True if new_coordinates[1] > previous_coordinates[1] + 10 else False
            previous_coordinates = new_coordinates
            nb_frames_ball_lost = 0
            trajectory.add_point(frame_idx + 1, previous_coordinates)
            if remove_keypoints:
                all_keypoints[frame_idx + 1].remove(new_coordinates)
        else:
            nb_frames_ball_lost +=1
        frame_idx += 1
    return trajectory


def filter_on_y_height(keypoints):
    """Filter keypoints that are probably too low to be a ball.
    
    Filters points out using the mean of y coordinates.
    :param list keypoints: list of keypoints for every frame.
    """
    y_coordinates = [point for points in keypoints if points != [] for point in list(zip(*points))[1]]
    mean = np.mean(y_coordinates)
    filtered_keypoints = []
    for frame_keypoints in keypoints:
        filtered_frame_keypoints = []
        for keypoint in frame_keypoints:
            if keypoint[1] < 1.4 * mean:
                filtered_frame_keypoints.append(keypoint)
        filtered_keypoints.append(filtered_frame_keypoints)
    return filtered_keypoints


def all_trajectories_checked(keypoints):
    """Checks if there are an keypoints left"""
    for frame in keypoints:
        if frame:
            return False
    return True


def first_non_empty_frame_idx(keypoints):
    """Returns the idex of the first frame that still has keypoints to look at."""
    for i, frame in enumerate(keypoints):
        if frame:
            return i


def keypoints_to_trajectory(keypoints):
    """
    
    """
    # First remove the keypoints that are probably too low to be a ball
    filtered_keypoints = filter_on_y_height(keypoints)
    trajectories = []
    while not all_trajectories_checked(filtered_keypoints):
        frame = first_non_empty_frame_idx(filtered_keypoints)
        trajectories.append(track_single_ball(filtered_keypoints, frame, 0, remove_keypoints=True))
    return trajectories


def filter_trajectories(trajectories):
    """

    :param trajectories:
    :return:
    """
    # For now just filter out short trajectories, could reintroduce the removed keypoints into other 
    # trajectories (could belong to another one)
    filtered_trajectories = [trajectory for trajectory in trajectories if len(trajectory.points) > 3]
    return filtered_trajectories


def trajectory_to_sparse_keypoints(trajectories, length):
    """

    :param trajectories:
    :param length:
    :return:
    """
    sparse_keypoints_list = [[] for _ in range(length)]
    for trajectory in trajectories:
        for keypoint in trajectory.points:
            sparse_keypoints_list[keypoint[0]].append(keypoint[1])
    return sparse_keypoints_list


## Lucas-Kanade method

def complete_trajectory_within_span(trajectory, lk_params, min_dist):
    """

    :param trajectory:
    :param lk_params:
    :param min_dist:
    :return:
    """
    for frame_idx in trajectory.span:
            if frame_idx in trajectory.found_frames:
                continue
            else:
                next_found_frame = trajectory.next_found_frame(frame_idx)
                temp_kps = []
                last_kp = trajectory.points_dict[frame_idx-1]
                for temp_frame_idx in range(frame_idx, next_found_frame+1):
                    opt_kps, st, err = cv2.calcOpticalFlowPyrLK(
                        frames[frame_idx-1],
                        frames[frame_idx],
                        np.array(last_kp, dtype=np.float32).reshape(-1, 1, 2),
                        None,
                        **lk_params
                    )
                    opt_kp = [opt_kp.ravel() if opt_kp is not None else None
                              for opt_kp in opt_kps][0]
                    if opt_kp is not None:
                        temp_kps.append((temp_frame_idx, opt_kp))
                        last_kp = opt_kp
                    else:
                        break
                next_found_kp = trajectory.points_dict[next_found_frame]
                if opt_kp is not None and np.sqrt(np.sum((np.array(next_found_kp)-opt_kp)**2)) < min_dist:
                    for index, kp in temp_kps:
                        trajectory.add_point(index, kp)
                else:
                    previous_kp = trajectory.points_dict[frame_idx-1]
                    slope = (np.array(next_found_kp) - np.array(previous_kp)
                            ) / (next_found_frame - frame_idx + 1)
                    for index in range(frame_idx, next_found_frame):
                        kp = previous_kp + slope*(index-frame_idx+1)
                        trajectory.add_point(index, kp)
    

def complete_trajectories(trajectories, frames, min_dist=10):
    """

    :param trajectories:
    :param frames:
    :param min_dist:
    :return:
    """
    lk_params = dict(
        winSize  = (100,100),
        maxLevel = 4,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    for trajectory in trajectories:
        complete_trajectory_within_span(trajectory, lk_params, min_dist)


def draw_trajectory(frames, trajectory, color=None, thickness=None):
    """

    :param frames:
    :param trajectory:
    :param color:
    :param thickness:
    :return:
    """
    if color is None:
        color = np.random.randint(0, 256, size=3).tolist()
    if thickness is None:
        thickness = frames[0].shape[0]//100
    for frame in trajectory.span:
        points = [(int(pt[1][0]), int(pt[1][1]))
                  for pt in sorted(trajectory.points)[:frame-trajectory.start_frame+1]]
        for point_idx in range(len(points)-1):
            cv2.line(frames[frame], points[point_idx], points[point_idx+1], color=color, thickness=thickness)
        cv2.circle(frames[frame], (int(points[-1][0]), int(points[-1][1])),
                   frames[0].shape[0]//70, color=color, thickness=-1)
    return frames



def draw_trajectories(frames, trajectories, color=None):
    """

    :param frames:
    :param trajectories:
    :param fps:
    :return:
    """
    displays = np.array([frame.copy() for frame in frames])
    if color is None:
        color = np.random.randint(0, 256, size=3).tolist()
    for trajectory in trajectories:
        displays = draw_trajectory(displays, trajectory, color=color)
    return displays


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        dest='file', type=str, help='Path to video file.'
    )
    parser.add_argument(
        "--start-frame", type=int, default=0,
        help="Frame to start detecting from."
    )
    parser.add_argument(
        "--end-frame", type=int, default=None,
        help="Frame to end detection on."
    )
    parser.add_argument(
        "--movement-path", type=str, default=None,
        help="Path to save the result of the movement extraction into. the movement video will not be saved by default"
    )
    parser.add_argument(
        "--keypoints-path", type=str, default=None,
        help="Path to save the result of the keypoints extraction into. the keypoints video will not be saved by default"
    )
    parser.add_argument(
        "--trajectories-path", type=str, default=None,
        help="Path to save the result of the trajectory results into. the trajectory video will not be saved by default"
    )
    parser.add_argument(
        "--save-fps", type=int, default=26,
        help="fps to use if saving as mp4."
    )
    parser.add_argument(
        "--display-fps", type=int, default=None,
        help="Display speed in fpm. If given when using the output-path argument, the output is displayed as well as saved."
    )
    parser.add_argument(
        "--keypoints-method", type=str, default="blob",
        help="(blob|hough): method to use to extract keypoints. defaults to blob"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    ## Get video
    print("Getting video")
    frames = np.array(utils.read_video(args.file))

    if args.end_frame is None:
        args.end_frame = len(frames)
    frames = frames[args.start_frame: args.end_frame]

    print("Running detection")
    ## Remove background
    foregrounds = get_foreground_masks(frames)

    ## Movement image cleaning
    # Erode and dilate to create nicer blobs
    frames_blob = [blobify(frame) for frame in foregrounds]

    if args.movement_path:
        print(f"Saving movements in {args.movement_path}")
        utils.save_video(frames_blob, args.movement_path, args.save_fps, color=False)

    ## Keypoints detection
    # Using Hough transform
    if args.keypoints_method == "hough":
        frames_keypoints = [get_keypoints_hough(frame) for frame in frames_blob]

    # Using blob detection
    else :
        detector = get_blob_detector()
        frames_keypoints = []  # blob coordinates in each frame
        for i, frame in enumerate(frames_blob):
            # Get keypoints for given frame
            keypoints = detector.detect(frame)
            # Add the frame’s keypoints to keypoint list
            frames_keypoints.append([keypoint.pt for keypoint in keypoints])

    ## Detect ball trajectories
    # Use blog coordinates to determine ball trajectories
    trajectories = keypoints_to_trajectory(frames_keypoints)

    # Fill trajectory gaps using the Lucas-Kanade method
    complete_trajectories(trajectories, frames, min_dist=50)

    # Filter trajectories
    filtered_trajectories = filter_trajectories(trajectories)

    # Determine final ball coordinates for each frame
    sparse_keypoints = trajectory_to_sparse_keypoints(
        filtered_trajectories,
        len(frames)
    )

    if args.keypoints_path:
        keypoints_display = [draw_keypoints(frame, kp, color=(0, 255, 0)) for frame, kp in zip(frames, sparse_keypoints)]
        print(f"Saving keypoints to path {args.keypoints_path}")
        utils.save_video(keypoints_display, args.keypoints_path, args.save_fps)

    trajectory_display = draw_trajectories(frames, filtered_trajectories, color=(0, 255, 0))

    if args.trajectories_path:
        print(f"Saving trajectories to path {args.trajectories_path}")
        utils.save_video(trajectory_display, args.trajectories_path, args.save_fps)

    display_speed = args.display_fps if args.display_fps else 26
    utils.display_frames(trajectory_display, fps=display_speed)
