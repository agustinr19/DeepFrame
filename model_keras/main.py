from model import *
from data import *

def test_depth_estimation():

    # data_loader = DataLoader("test_dataset")
    # train_samples = data_loader.training_set()
    # x_train = train_samples[:, :, :, :, 0]
    # y_train = np.repeat(train_samples[:, :, :, 0:2, 1], axis=3, repeats=8)
    # test_samples = data_loader.testing_set()
    # x_test = test_samples[:, :, :, :, 0]
    # y_test = np.repeat(test_samples[:, :, :, 0:2, 1], axis=3, repeats=8)

    # network_a = DenseSLAMNetSequential(frame_size=(256, 256, 3))

    network_b = DenseSLAMNet(frame_size=(256, 256, 3))

    # network_c = CNNSingle(frame_size=data_loader.frame_size())
    # network_c.train(x_train, x_test, y_train, y_test, epochs=10)

    # network_d = CNNStack(frame_size=(256, 256, 3))
    # network_d.train(x_train, x_test, y_train, y_test, epochs=10)    

if __name__ == "__main__":
    test_depth_estimation()