from model import *

def test_depth_estimation():
    network_a = DenseSLAMNet(frame_size=(256, 256, 3))
    network_b = CNNSingle(frame_size=(256, 256, 3))
    network_c = CNNStack(frame_size=(256, 256, 3))

if __name__ == "__main__":
    test_depth_estimation()