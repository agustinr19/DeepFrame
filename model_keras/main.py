from model import *
from data import *

def train_depth_estimation():

    data_loader = DataLoaderRGBD("../test_data/rgbd-scenes/kitchen_small", assemble_into_stacks=True, stack_length=5, training_split=0.9)
    train_samples = data_loader.training_set()

    # x_train = train_samples[:, :, :, :, 0]
    # y_train = np.repeat(train_samples[:, :, :, 0:2, 1], axis=3, repeats=8)
    test_samples = data_loader.testing_set()
    # x_test = test_samples[:, :, :, :, 0]
    # y_test = np.repeat(test_samples[:, :, :, 0:2, 1], axis=3, repeats=8)

    # training
    adjusted_frame_size = list(data_loader.frame_size())
    adjusted_frame_size[0] = ((adjusted_frame_size[0]//128)+1)*128
    adjusted_frame_size[1] = ((adjusted_frame_size[1]//128)+1)*128
    diff_x = adjusted_frame_size[0]-train_samples[0][0][0].shape[0]
    diff_y = adjusted_frame_size[1]-train_samples[0][0][0].shape[1]
    x_train = np.pad(train_samples[0], ((0, 0), (0, 0), (0, diff_x), (0, diff_y), (0, 0)), 'constant', constant_values=(0, 0))
    y_train = np.expand_dims(np.pad(train_samples[1], ((0, 0), (0, 0), (0, diff_x), (0, diff_y)), 'constant', constant_values=(0, 0)), axis=-1)
    x_test = np.pad(test_samples[0], ((0, 0), (0, 0), (0, diff_x), (0, diff_y), (0, 0)), 'constant', constant_values=(0, 0))
    y_test = np.expand_dims(np.pad(test_samples[1], ((0, 0), (0, 0), (0, diff_x), (0, diff_y)), 'constant', constant_values=(0, 0)), axis=-1)

    for n in range(x_train.shape[0]):
        print("Training set #"+str(n)+"...")
        network_a = DenseSLAMNetSequential(frame_size=adjusted_frame_size)
        network_a.train(x_train[n], x_test[0], y_train[n], y_test[0], epochs=1)

    # testing
    prediction = network_a.run(x_train[0])
    img = Image.fromarray(np.uint8(np.clip(prediction*100.0, 0.0, 254.0)))
    img.save("test.png")

    # network_b = DenseSLAMNet(frame_size=(256, 256, 3), frame_timespan=10)

    # network_c = CNNSingle(frame_size=data_loader.frame_size())
    # network_c.train(x_train, x_test, y_train, y_test, epochs=10)

    # network_d = CNNStack(frame_size=(256, 256, 3))
    # network_d.train(x_train, x_test, y_train, y_test, epochs=10)    

if __name__ == "__main__":
    train_depth_estimation()
