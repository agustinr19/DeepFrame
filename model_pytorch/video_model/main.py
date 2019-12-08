import sys, os, csv
import time, datetime
import numpy as np

from model import *
from dataloader import *

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import utils, criteria
from result import AverageMeter, Result

from visualize import save_visualization

### Adapted from img_model/main.py ###

args = utils.parse_args()

fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae',
                'delta1', 'delta2', 'delta3',
                'data_time', 'gpu_time']

def create_data_loaders():
    cnn_stack = args.cnn_type == 'stack'
    recurrent = args.recurrent == 'true'
    val_dataset_dir = os.path.join(args.base_dir, args.data_dir)

    val_dataset = CustomDataLoader(val_dataset_dir,train=False, stack=cnn_stack, stack_size=args.stack_size, concat=recurrent)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.n_workers)

    train_dataset_dir = os.path.join(args.base_dir, args.data_dir)
    train_dataset = CustomDataLoader(train_dataset_dir,train=True, stack=cnn_stack, stack_size=args.stack_size, concat=recurrent)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.n_workers)

    return (train, train_dataloader, val_dataloader)

def create_train_output_dir():
    output_dir = utils.fetch_output_dir(args)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.mkdir(os.path.join(output_dir, 'checkpoints'))
    print(output_dir)
    return output_dir

def create_val_output_dir():
    output_dir = utils.fetch_output_dir(args, sub_dir='val')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.mkdir(os.path.join(output_dir, 'visualizations'))

    return output_dir

def fetch_model_path():
    model_dir = os.path.join(utils.fetch_output_dir(args), 'checkpoints')
    models = [model for model in os.listdir(model_dir) if '.tar' in model]
    models = sorted(models)
    return os.path.join(model_dir, models[-1])

def print_and_write_csv(output_dir, result, train=True, epoch=None, iteration=None):
    csv_name = 'train.csv' if train else 'val.csv'
    csv_file = os.path.join(output_dir, csv_name)

    if train:
        str_output = 'Epoch: {:2d} Iteration: {:4d}, {}'.format(
            epoch, iteration, result.error_str()
        )
    else:
        print('{:.<80}'.format('Performing validation'))
        if isinstance(epoch, str):
            str_output = 'Epoch: {} {}'.format(epoch, result.error_metrics())
        else:
            str_output = 'Epoch: {:2d} {}'.format(epoch, result.error_metrics())

    with open(csv_file, 'a') as file:
        file.write('{}\n'.format(str_output))
        print(str_output)

def train_overview(train_dataloader, val_dataloader):
    global output_train, output_val

    output_dir = create_train_output_dir()
    output_train = os.path.join(output_dir, 'train.csv')
    output_val = os.path.join(output_dir, 'val.csv')

    with open(output_train, 'w') as train_csv:
        train_csv.write('{}\n'.format(','.join(fieldnames)))
    with open(output_val, 'w') as val_csv:
        val_csv.write('{}\n'.format(','.join(fieldnames)))

    print('Creating output dir {}'.format(output_dir))

    print("=> creating Model ...")
    if args.recurrent == 'true':
        model = DenseSLAMNet()
    elif args.cnn_type == 'single':
        model = CNN_Single()
    else:
        model = CNN_Stack()

    print("=> model created.")
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, \
        betas=args.betas,amsgrad=False)

    model = model.cuda()

    # define loss function (criterion) and optimizer
    if args.criterion == 'l2':
        criterion = criteria.MaskedL2Loss().cuda()
    elif args.criterion == 'l1':
        criterion = criteria.MaskedL1Loss().cuda()
    elif args.criterion == 'l1sum':
        criterion = criteria.L1LossSum()

    for epoch in range(args.n_epochs):
        train(train_dataloader, model, criterion, optimizer, epoch) # train for one epoch
        result = validate(val_dataloader, model) # evaluate on validation set

        utils.save_checkpoint({
            'args': args,
            'epoch': epoch,
            'model': model,
            'optimizer' : optimizer,
        }, epoch, output_dir)

def process_output(pred):
#    print(pred.shape)
    new = None
    for i in range(16):
        layer = pred[:,i,:,:]*(2**i)
#        print(layer.shape)
        layer = layer.view(pred.shape[0],1,pred.shape[2],pred.shape[3])
#        print(layer.shape)
        if new is None:
            new = layer
        else:
#            print(new.shape)
            new += layer
    return new

def train(train_dataloader, model, criterion, optimizer, epoch):
    average_meter = AverageMeter()
    model.train() # switch to train mode
    end = time.time()
    for i, (input, target) in enumerate(train_dataloader):

        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        print("TEST A: "+str(input.shape)+", "+str(target.shape))

        # compute pred
        end = time.time()
        pred = model(input)
#        pred = process_output(pred)

        print("TEST B")

        loss = criterion(pred, target)

        print("TEST C")

        optimizer.zero_grad()
        loss.backward() # compute gradient and do SGD step
        optimizer.step()
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        print("TEST D")

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        print("TEST E")

        if (i + 1) % args.print_freq == 0:
            print('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Loss{loss:.3f} '.format(
                  epoch, i+1, len(train_dataloader), data_time=data_time,
                  gpu_time=gpu_time, result=result, average=average_meter.average(),
                  loss = loss))

    avg = average_meter.average()
    with open(output_train, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel,
            'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
            'gpu_time': avg.gpu_time, 'data_time': avg.data_time})

def validate_overview(val_dataloader):
    image_dir = create_val_output_dir()
    model_path = fetch_model_path()

    state_dict = torch.load(model_path)
    model = state_dict['model']
    if args.criterion == 'l2':
        criterion = criteria.MaskedL2Loss()
    elif args.criterion == 'l1':
        criterion = criteria.MaskedL1Loss()
    elif args.criterion == 'l1sum':
        criterion = criteria.L1LossSum()

    validate(val_dataloader, model, save_image=True, image_dir=image_dir)

def validate(val_dataloader, model, write_to_file=True, save_image=False, image_dir=None):
    average_meter = AverageMeter()
    model.eval() # switch to evaluate mode
    end = time.time()
    for i, (input, target) in enumerate(val_dataloader):
        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred = model(input)
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if save_image:
            save_visualization(image_dir, i, input, pred, target)

        if (i+1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                   i+1, len(val_dataloader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if write_to_file:
        with open(output_val, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                'data_time': avg.data_time, 'gpu_time': avg.gpu_time})
    return avg

def main():
    train, train_dataloader, val_dataloader = create_data_loaders()

    train_overview(train_dataloader, val_dataloader)
    validate_overview(val_dataloader)

if __name__ == '__main__':
    main()
