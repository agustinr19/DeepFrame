from model import *
from data import *
from dataloader import *
from PIL import Image

def test_depth_estimation():

    # creates dataloaders
    timespan = 5
    scene = '../data/rgbd-scenes/meeting_small'
    val_dataloader = RGBDDataGenerator(scene, timespan=timespan, train=False)

    # loads frame size & initial params
    timespan = 1
    frame_size = None
    for (rgb, depth) in val_dataloader:
        timespan = rgb.shape[1]
        frame_size = rgb.shape[2:]
        break

    network_b = DenseSLAMNet(frame_size=frame_size, frame_timespan=timespan)
    test_img = network_b.run(val_dataloader[0][0])
    test_im = Image.fromarray(test_img[0, :, :, 0])
    test_im.save("test.png")

if __name__ == "__main__":
    test_depth_estimation()
