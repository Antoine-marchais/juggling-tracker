# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: 'Python 3.6.9 64-bit (''juggle_tracker'': venv)'
#     language: python
#     name: python36964bitjuggletrackervenv5397e930a4e5478eaf4e831a4957fa9d
# ---

# %% [markdown]
# # Object tracking tests

# %% [markdown]
# ## Imports

# %%
import cv2
import numpy as np
import utils
from tqdm import tqdm
import time
# %matplotlib inline

# %%
frames = utils.read_video("./data/juggling_front.mp4")
utils.display_frames(frames)

# %% [markdown]
# ## Image differences

# %% [markdown]
# Dans un premier temps nous allons calculer la différence entre les images successives de la vidéo, ce qui permet de capturer les brusque variations d'intensité dans le temps:

# %%
frames_gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames][:100]
frames_gray = [cv2.GaussianBlur(frame, (5,5), 0) for frame in frames_gray]
utils.display_frames(frames_gray, fps=20)


# %%
def diff_thresh(img1, img2, threshold):
    res = np.zeros(img1.shape)
    res[np.abs(img1-img2) > threshold] = 255
    return res


# %% [markdown]
# Le résultat étant assez bruité, nous allons moyenner la différence :

# %%
def diff_thresh_moy(img1, img2, threshold, window_size):
    res = np.zeros(img1.shape)
    diff = np.abs(img1-img2)
    kernel = np.ones((window_size, window_size))/(window_size**2)
    moy_diff = utils.convolve(diff, kernel)
    res[np.abs(img1-img2) > threshold] = 255
    return res


# %% [markdown]
# Regroupons ces fonctions dans une fonction de différence d'image généralisée : 

# %%
def get_diff(img1, img2, mode="basic", window_size=5):
    new_img1 = np.float32(img1)
    new_img2 = np.float32(img2)
    diff = np.abs(new_img1-new_img2)
    if mode == "mean":
        kernel = np.ones((window_size, window_size))/(window_size**2)
        diff = utils.convolve(diff, kernel)
    elif mode == "mindiff":
        diff = utils.conv_min(diff, (window_size, window_size))
    return np.uint8(diff)

def remove_local_mean(frames, mode="basic", window_size=5, time_delta=5):
    pad_frames = []
    new_frames = []
    overflow = (time_delta-1)//2
    for i in range(overflow):
        pad_frames.append(frames[0])
    pad_frames = pad_frames + frames
    for i in range(overflow):
        pad_frames.append(frames[-1])
    for frame_i in tqdm(range(len(frames))):
        context_frames = np.array(pad_frames[frame_i:frame_i+overflow] + pad_frames[frame_i+overflow+1:frame_i+time_delta], dtype=np.float32)
        mean = np.sum(context_frames, axis=0)/(time_delta-1)
        frame_diff = np.abs(np.float32(frames[frame_i])-mean)
        if mode == "mean":
            kernel = np.ones((window_size, window_size))/(window_size**2)
            frame_diff = utils.convolve(frame_diff, kernel)
        elif mode == "mindiff":
            frame_diff = utils.conv_min(frame_diff, (window_size, window_size))
        new_frames.append(np.uint8(frame_diff))
    return new_frames
        


# %%
frames_diff = []

frames_diff = remove_local_mean(frames_gray, mode="basic", window_size=7, time_delta=5)

for i in tqdm(range(len(frames_gray)-2)):
    frames_diff.append(get_diff(frames_gray[i]/2, frames_gray[i+2]/2, mode="mindiff", window_size=7))
utils.display_frames(frames_diff)

utils.display_frames(frames_diff, fps=5)

# %%
frames_thresh = [cv2.threshold(frame, 10, 255, cv2.THRESH_BINARY)[1] for frame in frames_diff]
time.sleep(3)
utils.display_frames(frames_thresh)


# %% [markdown]
# ## Background removal

# %%
def get_foreground_masks(frames):
    backsub = cv2.createBackgroundSubtractorMOG2(history=3, varThreshold=10)
    # training the substractor
    for frame in tqdm(frames[:30], desc="training substractor : "):
        backsub.apply(frame)

    # getting foreground masks
    foregrounds = []
    for frame in tqdm(frames, desc="applying subtractor : ") :
        foregrounds.append(backsub.apply(frame))
    return [cv2.threshold(frame, 10, 255, cv2.THRESH_BINARY)[1] for frame in foregrounds]

foregrounds = get_foreground_masks(frames[:100])


# %%
utils.display_frames(foregrounds, fps=5)

# %% [markdown]
# ## Extracting movement with optical flow

# %%
optical_flows = []
for i_frame in tqdm(range(len(frames_gray)-1)):
    hsv_flow = np.zeros_like(frames[0])
    flow = cv2.calcOpticalFlowFarneback(frames_gray[i_frame], frames_gray[i_frame+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv_flow[...,1] = 255
    hsv_flow[...,0] = ang*180/np.pi/2
    hsv_flow[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr_flow = cv2.cvtColor(hsv_flow,cv2.COLOR_HSV2BGR)
    optical_flows.append(bgr_flow)

# %%
utils.display_frames(optical_flows, fps=10)


# %% [markdown]
# ## Circle detection with hough transform

# %%
def expand_foreground(img, it):
    res = img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    res = cv2.erode(res, kernel, 15)
    res = cv2.dilate(res, kernel, iterations=it)
    return res


# %%
results = [np.copy(frame) for frame in frames[:100]]

for display, frame in zip(results, frames_blob):
    # detect circles in the image
    circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, dp=4, minDist=100, param1=100, param2=60, minRadius=10, maxRadius=50)
    # ensure at least some circles were found
    if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(display, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(display, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            # show the output image

# %%
time.sleep(3)
utils.display_frames(results, fps=10)

# %%
cv2.imwrite("./data/report/hough_circles.png", results[10][:, 500:1350])


# %% [markdown]
# ## blob detection

# %%
def blobify(img):
    kernel_erode = np.ones((3,3), dtype=np.uint8)
    kernel_dilate = np.ones((3,3), dtype=np.uint8)
    res = img.copy()
    for i in range(4):
        res= cv2.erode(res, kernel_erode, iterations=2)
        res = cv2.dilate(res, kernel_dilate, iterations=2)
    return res

frames_blob = [blobify(frame) for frame in foregrounds]

# %%
cv2.imwrite("./data/report/erode_dilate.png", frames_blob[10][:, 500:1350])

# %%
utils.display_frames(frames_blob, fps=5)

# %%
params = cv2.SimpleBlobDetector_Params()

params.filterByColor = True
params.blobColor = 255

params.filterByCircularity = True
params.minCircularity = 0.1

params.filterByArea = True
params.minArea = 300
params.maxArea = 1000000

params.filterByConvexity = False

params.filterByInertia = True
params.minInertiaRatio = 0.4

params.minDistBetweenBlobs = 70

index_img = 20

detector = cv2.SimpleBlobDetector_create(params)
frames_keypoints = []

def draw_keypoints(img, keypoints, color=None, radius=None):
    if color == None:
        color=np.random.randint(0,256,size=3).tolist()
    if radius == None:
        radius = img.shape[0]//70
    result = np.copy(img)
    for keypoint in keypoints:
        cv2.circle(result, (int(keypoint[0]), int(keypoint[1])), radius, color=color, thickness=-1)
    return result

results = []
for i, frame in enumerate(frames_blob):
    keypoints = detector.detect(frame)
    frames_keypoints.append([keypoint.pt for keypoint in keypoints])
    results.append(draw_keypoints(frames[i], frames_keypoints[i], color=(0, 255, 0)))


# %%
def draw_blobs(img, blobs, keypoints, min_dist=70):
    res = img.copy()
    contour, hier = cv2.findContours(blobs,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        if len(keypoints) > 0:
            M = cv2.moments(cnt)
            center = np.array([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])
            dists = np.array([np.sum((np.array(kp)-center)**2)**0.5 for kp in keypoints])
            if np.min(dists) <= min_dist :
                cv2.drawContours(res,[cnt],0,[10, 180, 10],-1)
    res = draw_keypoints(res, keypoints, color=(0, 0, 200))
    return res


# %%
blobs = [draw_blobs(frame, blob, kp) for (frame, blob, kp) in zip(frames, frames_blob, frames_keypoints)]

# %%
utils.display_frames(blobs, fps=10)

# %%
cv2.imwrite("./data/report/blobs.png", blobs[10][:, 500:1350])

# %%
keypoints[0].size

# %%
utils.display_frames(results, fps=10)


# %% [markdown]
# ## Finding trajectories with Lucas-Kanade method

# %%
class Trajectory:
    def __init__(self, frame_i, startpoint):
        self.start_frame = frame_i
        self.points = [startpoint]
        self.staleness = 0

def get_trajectories(frames, frames_kp, min_dist=10, max_staleness=1):
    lk_params = dict( winSize  = (100,100),
                    maxLevel = 4,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    saved_trajectories = []
    current_trajectories = [Trajectory(0,kp) for kp in frames_kp[0]]
    for i in range(1,len(frames)):

        # we compute optical flow for the points in the last trajectories
        last_kps = np.array([trajectory.points[-1] for trajectory in current_trajectories], dtype=np.float32).reshape(-1,1,2)
        opt_kps, st, err = cv2.calcOpticalFlowPyrLK(frames[i-1], frames[i], last_kps, None, **lk_params)
        opt_kps = [opt_kps[j].ravel() for j in range(opt_kps.shape[0])] if opt_kps is not None else []

        # we extend the trajectories with the coresponding keypoint, or the calculated optical flow
        extend_trajectories(current_trajectories, opt_kps, frames_kp[i], min_dist, i)
        
        # we save trajectories which have gone stale
        stale_trajectories = remove_stale_trajectories(current_trajectories, max_staleness)
        saved_trajectories = saved_trajectories + stale_trajectories
    cv2.destroyAllWindows()
    return saved_trajectories

def extend_trajectories(trajectories, optical_keypoints, real_keypoints, min_dist, frame_i):
    added_kps = []
    for trajectory, opt_kp in zip(trajectories, optical_keypoints):
        added_to_trajectory = False
        for kp in real_keypoints:
            if np.sum((np.array(kp)-opt_kp)**2) < min_dist and not added_to_trajectory and not kp in added_kps:
                trajectory.staleness = 0
                trajectory.points.append(kp)
                added_kps.append(kp)
                added_to_trajectory = True
        if not added_to_trajectory :
            trajectory.staleness += 1
            trajectory.points.append(opt_kp)
    remaining_kps = [kp for kp in real_keypoints if kp not in added_kps]
    for kp in remaining_kps :
        trajectories.append(Trajectory(frame_i, kp))

def remove_stale_trajectories(trajectories, max_staleness):
    stale_trajectories = [trajectory for trajectory in trajectories if trajectory.staleness >= max_staleness]
    for trajectory in stale_trajectories:
        trajectories.remove(trajectory)
    return stale_trajectories
            
trajectories = get_trajectories(frames[:100], frames_keypoints, min_dist=200, max_staleness=3)

# %%

# %%
print(max([len(trajectory.points) for trajectory in trajectories]))
print(len(trajectories))
print(sum([len(kps) for kps in frames_keypoints]))

# %%
best_trajectories = sorted(trajectories, key=lambda trajectory:len(trajectory.points), reverse=True)[:20]
print(best_trajectories[0].points)


# %%
def draw_trajectory(img, trajectory, color=None, thickness=None):
    if color == None:
        color = np.random.randint(0, 256, size=3).tolist()
    if thickness == None:
        thickness = img.shape[0]//100
    res = img.copy()
    points = [(int(pt[0]), int(pt[1])) for pt in trajectory.points]
    for i in range(len(points)-1):
        cv2.line(res, points[i], points[i+1], color=color, thickness=thickness)
    return res

trajectory = best_trajectories[5]
print(trajectory.start_frame)
img_trajectory = draw_trajectory(frames[trajectory.start_frame], trajectory)
utils.display_img(img_trajectory)


# %%
def view_trajectories(frames, trajectories, fps=5):
    displays = [frame.copy() for frame in frames]
    colors = [np.random.randint(0, 256, size=3).tolist() for trajectory in trajectories]
    for i in range(len(frames)):
        for color, trajectory in zip(colors, trajectories):
            if i >= trajectory.start_frame and i < trajectory.start_frame + len(trajectory.points):
                displays[i] = draw_trajectory(displays[i], trajectory, color=color)
                trajectory_point = trajectory.points[i-trajectory.start_frame]
                cv2.circle(displays[i], (int(trajectory_point[0]), int(trajectory_point[1])), frames[i].shape[0]//70, color=color, thickness=-1)
    utils.display_frames(displays, fps=fps)

view_trajectories(frames[:100], best_trajectories, fps=5)

# %%
