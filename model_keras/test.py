from model import *
from data import *
from dataloader import *

def test_depth_estimation():

    # creates dataloaders
    timespan = 5
    scene = '../data/rgbd-scenes/meeting_small'
    val_dataloader = RGBDDataGenerator(scene, timespan=timespan, train=False)

    # loads frame size & initial params
    timespan = 1
    frame_size = None
    for (rgb, depth) in train_dataloader:
        timespan = rgb.shape[1]
        frame_size = rgb.shape[2:]
        break

    network_b = DenseSLAMNet(frame_size=frame_size, frame_timespan=timespan)
    test_img = network_b.run(val_dataloader[0])
    print(test_img.shape)

if __name__ == "__main__":
    train_depth_estimation()
