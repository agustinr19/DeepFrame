from model import *
from data import *
from dataloader import *

def train_depth_estimation():

    # data_loader = DataLoaderRGBD("../test_data/rgbd-scenes/table", assemble_into_stacks=True, stack_length=5, training_split=0.8)

    # creates dataloaders
    scene = '../data/rgbd-scenes/background'
    train_dataloader = RGBDDataGenerator(scene, train=True)
    val_dataloader = RGBDDataGenerator(scene, train=False)

    # loads frame size & initial params
    timespan = 1
    frame_size = None
    for (rgb, depth) in train_dataloader:
        timespan = rgb.shape[0]
        frame_size = rgb.shape[1:]
        break

    # loads data into samples
    # train_samples = data_loader.training_set()
    # x_train = train_samples[0]
    # y_train = train_samples[1]
    # test_samples = data_loader.testing_set()
    # x_test = test_samples[0]
    # y_test = test_samples[1]

    # training
    # adjusted_frame_size = list(data_loader.frame_size())
    # adjustment_divisor = 128
    # adjusted_frame_size[0] = ((adjusted_frame_size[0]//adjustment_divisor)+1)*adjustment_divisor
    # adjusted_frame_size[1] = ((adjusted_frame_size[1]//adjustment_divisor)+1)*adjustment_divisor
    # diff_x = adjusted_frame_size[0]-x_train.shape[2]
    # diff_y = adjusted_frame_size[1]-x_train.shape[3]
    # x_train = np.pad(x_train, ((0, 0), (0, 0), (0, diff_x), (0, diff_y), (0, 0)), 'constant', constant_values=(0, 0))
    # y_train = np.expand_dims(np.pad(y_train, ((0, 0), (0, 0), (0, diff_x), (0, diff_y)), 'constant', constant_values=(0, 0)), axis=-1)
    # x_test = np.pad(x_test, ((0, 0), (0, 0), (0, diff_x), (0, diff_y), (0, 0)), 'constant', constant_values=(0, 0))
    # y_test = np.expand_dims(np.pad(y_test, ((0, 0), (0, 0), (0, diff_x), (0, diff_y)), 'constant', constant_values=(0, 0)), axis=-1)

    # for n in range(x_train.shape[0]):
    #     print("Training set #"+str(n)+" of "+str(x_train.shape[0])+"...")
    #     network_a = DenseSLAMNetSequential(frame_size=adjusted_frame_size)
    #     network_a.train(x_train[n], x_test[0], y_train[n], y_test[0], epochs=1)
    #     del network_a

    # testing
    # network_a = DenseSLAMNetSequential(frame_size=adjusted_frame_size)
    # input_im = np.expand_dims(x_train[0][0], axis=0)
    # prediction = network_a.run(input_im)[0, :, :, 0]
    # img = Image.fromarray(np.uint8(np.clip(prediction*100.0, 0.0, 254.0)))
    # img.save("test.png")

    # frame_size = (480, 640, 3)
    network_b = DenseSLAMNet(frame_size=frame_size, frame_timespan=5)
    network_b.train_with_dataloaders(train_dataloader, val_dataloader)

    # network_c = CNNSingle(frame_size=data_loader.frame_size())
    # network_c.train(x_train, x_test, y_train, y_test, epochs=10)

    # network_d = CNNStack(frame_size=(256, 256, 3))
    # network_d.train(x_train, x_test, y_train, y_test, epochs=10)    

if __name__ == "__main__":
    train_depth_estimation()
