import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(path, color):
    """Load an image in a numpy array
    
    Arguments:
        path {string} -- path of the image
        color {bool} -- boolean for the image being in color
    
    Returns:
        ndarray -- 2D or 3D array of the image
    """
    if color:
        img = cv2.imread(path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(path, 0)
    return img

def display_img(img):
    """Plot the image in a new figure
    
    Arguments:
        img {ndarray} -- array containing image
    """
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot()
    if len(img.shape) == 3:
        ax.imshow(img, interpolation="nearest")
    else:
        ax.imshow(img, interpolation="nearest", cmap='gray')

def scale_to_screen(img):
    """Resize the image to fit into a 800x600 screen

    Arguments:
        img {ndarray} -- image to resize

    Returns:
        ndarray -- resized image
    """
    height = img.shape[0]
    width = img.shape[1]
    if height*4/3 > width :
        scale_factor = height/600
    else :
        scale_factor = width/800
    width = int(width/scale_factor)
    height = int(height/scale_factor)
    return cv2.resize(img, (width, height))

def read_video(path):
    """Read all frames of the video into a list of images

    Arguments:
        path {string} -- path to the video

    Returns:
        list(ndarray) -- list of all the frames
    """
    cap = cv2.VideoCapture(path)
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret :
            frames.append(frame)
        else :
            cap.release()
    return frames

def display_frames(frames, fps=50):
    """Display the sequence of frames using the fps setting

    Arguments:
        frames {list(ndarray)} -- list of frames in the video

    Keyword Arguments:
        fps {int} -- number of frames per second (default: {50})
    """
    for frame in frames:
        resized = scale_to_screen(frame)
        cv2.imshow(f"frame",resized)
        cv2.waitKey(1000//fps)
    cv2.destroyAllWindows()

def convolve(I,h,zero_padding=True):
    """Computes the convolution of image I by mask h
    
    Arguments:
        I {ndarray} -- image
        h {ndarray} -- convolution kernel
    
    Keyword Arguments:
        zero_padding {bool} -- wether to use 0 padding (default: {True})
    
    Returns:
        ndarray -- convolved image
    """
    overflow_row = (h.shape[0]-1)//2
    overflow_col = (h.shape[1]-1)//2
    if zero_padding:
        accum = np.zeros((I.shape[0]+2*overflow_row, I.shape[1]+2*overflow_col))
    for row in range(h.shape[0]):
        for col in range(h.shape[1]):
            accum[row:row+I.shape[0],col:col+I.shape[1]] += h[row,col]*I
    return accum[overflow_row:-overflow_row,overflow_col:-overflow_col]

def conv_min(I, shape):
    """Computes the minimum of the image I in the neighbourhoud shape

    Arguments:  
        I {ndarray} -- image
        shape {(int,int)} -- neighboorhood to consider

    Keyword Arguments:
        zero_padding {bool} -- wether to use 0 padding (default: {True})

    Returns:
        ndarray -- image with min of each neigbourhood
    """
    overflow_row = (shape[0]-1)//2
    overflow_col = (shape[1]-1)//2
    accum = np.max(I) * np.ones(I.shape)
    for row in range(shape[0]):
        for col in range(shape[1]):
            row_slice_img = [max(row-overflow_row,0),I.shape[0]-max(overflow_row-row,0)]
            col_slice_img = [max(col-overflow_col,0),I.shape[1]-max(overflow_col-col,0)]
            row_slice_accum = [max(overflow_row-row,0), I.shape[0]+min(overflow_row-row,0)]
            col_slice_accum = [max(overflow_col-col,0), I.shape[1]+min(overflow_col-col,0)]
            accum[row_slice_accum[0]:row_slice_accum[1], col_slice_accum[0]:col_slice_accum[1]] = np.minimum(
                accum[row_slice_accum[0]:row_slice_accum[1], col_slice_accum[0]:col_slice_accum[1]], 
                I[row_slice_img[0]:row_slice_img[1], col_slice_img[0]:col_slice_img[1]]
            )
    return accum

def conv_max(I, shape):
    """Computes the maximum in the neighbourhood of each point

    Arguments:
        I {ndarray} -- image
        shape {(int,int)} -- neighboorhood to consider

    Returns:
        ndarray -- image with max of each neighbourhood
    """
    return np.max(I)-conv_min(np.max(I)-I,shape)

def get_gaussian_kernel(size,sigma=None):
    """get the gaussian kernel of given size

    Arguments:
        size {int} -- size of the kernel
        sigma {float} -- standard deviation of the kernel, deduced from the size if ommited

    Returns:
        ndarray -- gaussian kernel
    """
    kernel = np.zeros((size,size))
    if sigma==None:
        sigma =  0.3*((size-1)*0.5 - 1) + 0.8 
    for i in range(size):
        for j in range(size):
            x = i-(size-1)/2
            y = j-(size-1)/2
            kernel[i,j] = np.exp(-(x**2+y**2)/(2*sigma**2))
    return kernel/np.sum(kernel)