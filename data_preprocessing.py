import os
import subprocess
from pytube import YouTube 

CWD = os.getcwd() #current working directory

def load_frames(frame_folder_path):
  """
  Loads a set of frames for machine learning applications

  Args:
      frame_folder_path (str): Path to video specific frame folder

  Returns:
      Tensor of frame data
  """
  pass

def generate_depth_map(image_path):
  """
  Geometrically generates a depth map from an image

  Args:

  Returns:

  """
  pass

def video_to_frames(video_path,frame_path,fps):
    """
    Converts a video file to a set of images based on a given frame rate. 
    (Needs ffmpeg to be installed and added to PATH environment to use commandline functions)

    Args:
        video_path (str): Path to video file
        frame_path (str): Path to folder where images will be saved. 
                          A new folder will be created for each video.
        fps (int): frames per second for frame extraction

    Returns:
        Subprocess response to ffmpeg commandline operations
    """
    name = video_path.split('\\')[-1].split('.')[0] 

    # print(name,video_path)
    if not os.path.exists(frame_path):
            os.makedirs(frame_path)    

    if not os.path.exists(frame_path + name):
            os.makedirs(frame_path + name)
    
    query = "ffmpeg -i " + video_path + " -vf fps=" + str(fps) + " " + frame_path + name + "/output%06d.png"
    response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()
    s = str(response).encode('utf-8')

    return s

def download_youtube_video(link,video_path,video_name):
    """
    Downloads a video from YouTube.

    Args:
        link (str): URL for YouTube video
        video_path (str): Path where the video will be saved
        video_name (str): Name of video file (if not given, file will default to original name)
    """ 
    if not os.path.exists(video_path):
        os.makedirs(video_path)

    try: 
        yt = YouTube(link) 
    except: 
        print("Could not find video. Check your internet conncention.") 
      
    mp4files = yt.streams.filter(subtype='mp4') 
    yt.set_filename(video_name) 
    video_download = mp4files.first() #download first option (TODO modify for selecting resolution)
    
    try: 
        video_download.download(video_path) 
    except: 
        print("Could not download video. Check your internet conncention.") 

    
# TEST CODE 
# download test video from youtube
download_youtube_video('https://www.youtube.com/watch?v=kacyaEXqVhs',CWD+'\\videos\\','test') 
# use shorter video clip for testing purposes
query = "ffmpeg -i " + CWD +"\\videos\\test.mp4 -ss 00:00:05 -t 00:01:05 -async 1 " + CWD +"\\videos\\test_clip.mp4"
response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()
s = str(response).encode('utf-8')
# extract frames
video_to_frames(CWD+'\\videos\\'+'test_clip.mp4',CWD+'\\imgs\\',3) 
