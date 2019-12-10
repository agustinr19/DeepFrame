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

    test_in = Image.fromarray(np.uint8(val_dataloader[0][0][0, -1, :, :, :]))
    test_in.save("test_in.png")

    print(val_dataloader[0][1].shape)
    test_ref = Image.fromarray(np.uint8(val_dataloader[0][1][0, -1, :, :, :]))
    test_ref.save("test_ref.png")

    test_out = Image.fromarray(np.uint8(test_img[0, :, :, 0]*255))
    test_out.save("test_out.png")

if __name__ == "__main__":
    test_depth_estimation()
