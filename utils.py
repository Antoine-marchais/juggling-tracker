import cv2

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