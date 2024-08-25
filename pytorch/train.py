import pathlib
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np
import os
import cv2

from utils import *
from options import parse_args

from models.model import Model

from datasets.floorplan_dataset import FloorplanDataset
from IP import reconstructFloorplan

# Example usage for prediction
# python train.py --task=predict --prediction_dir ./prediction_inputs/
# create a directory with your images for prediction and execute the above command in the terminal on the pytorch directory


def main(options):
    if not os.path.exists(options.checkpoint_dir):
        os.system("mkdir -p %s"%options.checkpoint_dir)
        pass
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass

    # Set the device based on the --cpu flag
    device = torch.device('cpu' if options.cpu else 'cuda')

    if(options.task == 'predict'):
        predictForDataset(options)
        exit(1)

    dataset = FloorplanDataset(options, split='train', random=True)

    print('the number of images', len(dataset))    

    dataloader = DataLoader(dataset, batch_size=options.batchSize, shuffle=True, num_workers=16)

    model = Model(options)
    if options.cpu:
         model.to(device)
    else:
        model.cuda()
        
    model.train()

    if options.restore == 1:
        print('restore')
        model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint.pth'))
        pass

    
    if options.task == 'test':
        dataset_test = FloorplanDataset(options, split='test', random=False)
        testOneEpoch(options, model, dataset_test)
        exit(1)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = options.LR)
    if options.restore == 1 and os.path.exists(options.checkpoint_dir + '/optim.pth'):
        optimizer.load_state_dict(torch.load(options.checkpoint_dir + '/optim.pth'))
        pass

    for epoch in range(options.numEpochs):
        epoch_losses = []
        data_iterator = tqdm(dataloader, total=len(dataset) // options.batchSize + 1)
        for sampleIndex, sample in enumerate(data_iterator):
            optimizer.zero_grad()
            
            images, corner_gt, icon_gt, room_gt = sample[0].to(device), sample[1].to(device), sample[2].to(device), sample[3].to(device)

            corner_pred, icon_pred, room_pred = model(images)
            #print([(v.shape, v.min(), v.max()) for v in [corner_pred, icon_pred, room_pred, corner_gt, icon_gt, room_gt]])
            #exit(1)
            #print(corner_pred.shape, corner_gt.shape)
            #exit(1)
            corner_loss = torch.nn.functional.binary_cross_entropy(corner_pred, corner_gt)
            icon_loss = torch.nn.functional.cross_entropy(icon_pred.view(-1, NUM_ICONS + 2), icon_gt.view(-1))
            room_loss = torch.nn.functional.cross_entropy(room_pred.view(-1, NUM_ROOMS + 2), room_gt.view(-1))            
            losses = [corner_loss, icon_loss, room_loss]
            loss = sum(losses)

            loss_values = [l.data.item() for l in losses]
            epoch_losses.append(loss_values)
            status = str(epoch + 1) + ' loss: '
            for l in loss_values:
                status += '%0.5f '%l
                continue
            data_iterator.set_description(status)
            loss.backward()
            optimizer.step()

            if sampleIndex % 500 == 0:
                visualizeBatch(options, images.detach().cpu().numpy(), [('gt', {'corner': corner_gt.detach().cpu().numpy(), 'icon': icon_gt.detach().cpu().numpy(), 'room': room_gt.detach().cpu().numpy()}), ('pred', {'corner': corner_pred.max(-1)[1].detach().cpu().numpy(), 'icon': icon_pred.max(-1)[1].detach().cpu().numpy(), 'room': room_pred.max(-1)[1].detach().cpu().numpy()})])
                if options.visualizeMode == 'debug':
                    exit(1)
                    pass
            continue
        print('loss', np.array(epoch_losses).mean(0))
        if True:
            torch.save(model.state_dict(), options.checkpoint_dir + '/checkpoint.pth')
            torch.save(optimizer.state_dict(), options.checkpoint_dir + '/optim.pth')
            pass

        #testOneEpoch(options, model, dataset_test)        
        continue
    return

def predictForDataset(options):
    
    device = torch.device('cpu' if options.cpu else 'cuda')
    num_workers: int = 1
    directory_prediction: pathlib.Path = options.prediction_dir
    prediction_file: pathlib.Path = directory_prediction.joinpath('predict.txt')
    if (not directory_prediction.exists()):
        raise ValueError('The prediction directory does not exist')
    
    images: list[pathlib.Path] = list(directory_prediction.glob('*.png'))
    images.extend(directory_prediction.glob('*.jpg'))

    if(not len(images)):
        print(f'Nothing to do as there are no images inside the directory {directory_prediction}')
        exit(1)
    
    images.sort()

    f = open(prediction_file, 'w')
    for im_path in images:
        blank_annotation_file: pathlib.Path = im_path.parent.joinpath(f'{im_path.stem}.txt')
        annotation_file = open(blank_annotation_file, 'w')
        annotation_file.close()
        f.write(f'{im_path.stem}{im_path.suffix}\t{blank_annotation_file.stem}{blank_annotation_file.suffix}\n')
    f.close()

    options.dataFolder = directory_prediction
    options.batchSize = 1

    dataset = FloorplanDataset(options, split='predict', random=False)

    model = Model(options)
    if options.cpu:
        print("load cpu device")
        model.to(device)
        model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint.pth', map_location=torch.device('cpu')))
    else:
        print("load cuda device")
        model.cuda()
        model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint.pth'))
        
    model.eval()

    dataloader = DataLoader(dataset, batch_size=options.batchSize, shuffle=False, num_workers=num_workers)
    data_iterator = tqdm(dataloader, total=len(dataset) // options.batchSize + 1)

    for sampleIndex, sample in enumerate(data_iterator):
        wallInformationFile: pathlib.Path = pathlib.Path('./test/').joinpath('floorplan').joinpath(f'{sampleIndex}_0_floorplan.txt')
        wallInformationDrawing: pathlib.Path = pathlib.Path('./test/').joinpath('floorplan').joinpath(f'{sampleIndex}_0_floorplan_drawing.jpg')
        images, corner_gt, icon_gt, room_gt = sample[0].to(device), sample[1].to(device), sample[2].to(device), sample[3].to(device)       
        corner_pred, icon_pred, room_pred = model(images)
        
        if(not wallInformationFile.exists()):
            print('run prediction algorithm')
            visualizeBatch(options, 
                        images.detach().cpu().numpy(), 
                        [
                            (
                                'gt', 
                                {
                                'corner': corner_gt.detach().cpu().numpy(), 
                                'icon': icon_gt.detach().cpu().numpy(), 
                                'room': room_gt.detach().cpu().numpy()
                                }), 
                            (
                                'pred', 
                            {
                                'corner': corner_pred.max(-1)[1].detach().cpu().numpy(), 
                                'icon': icon_pred.max(-1)[1].detach().cpu().numpy(), 
                                'room': room_pred.max(-1)[1].detach().cpu().numpy()
                                })
                                ], 0, f'{sampleIndex}_')            
            for batchIndex in range(len(images)):
                    corner_heatmaps = corner_pred[batchIndex].detach().cpu().numpy()
                    icon_heatmaps = torch.nn.functional.softmax(icon_pred[batchIndex], dim=-1).detach().cpu().numpy()
                    room_heatmaps = torch.nn.functional.softmax(room_pred[batchIndex], dim=-1).detach().cpu().numpy()                
                    reconstructFloorplan(corner_heatmaps[:, :, :NUM_WALL_CORNERS], 
                                        corner_heatmaps[:, :, NUM_WALL_CORNERS:NUM_WALL_CORNERS + 4], 
                                        corner_heatmaps[:, :, -4:], 
                                        icon_heatmaps, 
                                        room_heatmaps, 
                                        output_prefix=options.test_dir + '/' + str(sampleIndex) + '_' + str(batchIndex) + '_', 
                                        densityImage=None, 
                                        gt_dict=None, 
                                        gt=False, 
                                        gap=-1, 
                                        distanceThreshold=-1, 
                                        lengthThreshold=-1, 
                                        debug_prefix='test', 
                                        heatmapValueThresholdWall=None, 
                                        heatmapValueThresholdDoor=None, 
                                        heatmapValueThresholdIcon=None, 
                                        enableAugmentation=True)

        if(wallInformationFile.exists()):
            wall_start_index = 2
            f = open(f'{wallInformationFile}', 'r')
            lines = f.read().split('\n')
            w, h = [int(val.strip()) for val in lines[0].split('\t')]
            total_walls = [int(val.strip()) for val in lines[1].split('\t')][0]

            drawing_image = np.zeros((h, w, 3), np.uint8)
            drawing_image[:,:,:,] = 255

            for i in range(total_walls):
                index = i + wall_start_index
                x1, y1, x2, y2, _, _ = [int(float(val.strip())) for val in lines[index].split('\t')]
                cv2.circle(drawing_image, (x1, y1), 5, (0, 0, 255), -1 )
                cv2.circle(drawing_image, (x2, y2), 5, (0, 0, 255), -1 )
                cv2.line(drawing_image, (x1, y1), (x2, y2), (255, 0, 0), 1)

            f.close()
            cv2.imwrite(f'{wallInformationDrawing}', drawing_image)

    # testOneEpoch(options, model, dataset)


def drawWalls():
    pass

def testOneEpoch(options, model, dataset):
    model.eval()
    
     # Set the device based on the --cpu flag
    device = torch.device('cpu' if options.cpu else 'cuda')

    dataloader = DataLoader(dataset, batch_size=options.batchSize, shuffle=False, num_workers=1)
    
    epoch_losses = []    
    data_iterator = tqdm(dataloader, total=len(dataset) // options.batchSize + 1)
    for sampleIndex, sample in enumerate(data_iterator):

        images, corner_gt, icon_gt, room_gt = sample[0].to(device), sample[1].to(device), sample[2].to(device), sample[3].to(device)
        
        corner_pred, icon_pred, room_pred = model(images)
        corner_loss = torch.nn.functional.binary_cross_entropy(corner_pred, corner_gt)
        icon_loss = torch.nn.functional.cross_entropy(icon_pred.view(-1, NUM_ICONS + 2), icon_gt.view(-1))
        room_loss = torch.nn.functional.cross_entropy(room_pred.view(-1, NUM_ROOMS + 2), room_gt.view(-1))            
        losses = [corner_loss, icon_loss, room_loss]
        
        loss = sum(losses)

        loss_values = [l.data.item() for l in losses]
        epoch_losses.append(loss_values)
        status = 'val loss: '
        for l in loss_values:
            status += '%0.5f '%l
            continue
        data_iterator.set_description(status)

        if sampleIndex % 500 == 0:
            visualizeBatch(options, images.detach().cpu().numpy(), [('gt', {'corner': corner_gt.detach().cpu().numpy(), 'icon': icon_gt.detach().cpu().numpy(), 'room': room_gt.detach().cpu().numpy()}), ('pred', {'corner': corner_pred.max(-1)[1].detach().cpu().numpy(), 'icon': icon_pred.max(-1)[1].detach().cpu().numpy(), 'room': room_pred.max(-1)[1].detach().cpu().numpy()})])            
            for batchIndex in range(len(images)):
                corner_heatmaps = corner_pred[batchIndex].detach().cpu().numpy()
                icon_heatmaps = torch.nn.functional.softmax(icon_pred[batchIndex], dim=-1).detach().cpu().numpy()
                room_heatmaps = torch.nn.functional.softmax(room_pred[batchIndex], dim=-1).detach().cpu().numpy()                
                reconstructFloorplan(corner_heatmaps[:, :, :NUM_WALL_CORNERS], corner_heatmaps[:, :, NUM_WALL_CORNERS:NUM_WALL_CORNERS + 4], corner_heatmaps[:, :, -4:], icon_heatmaps, room_heatmaps, output_prefix=options.test_dir + '/' + str(batchIndex) + '_', densityImage=None, gt_dict=None, gt=False, gap=-1, distanceThreshold=-1, lengthThreshold=-1, debug_prefix='test', heatmapValueThresholdWall=None, heatmapValueThresholdDoor=None, heatmapValueThresholdIcon=None, enableAugmentation=True)
                continue
            if options.visualizeMode == 'debug':
                exit(1)
                pass
        continue
    print('validation loss', np.array(epoch_losses).mean(0))

    model.train()
    return

def visualizeBatch(options, images, dicts, indexOffset=0, prefix=''):
    #cornerColorMap = {'gt': np.array([255, 0, 0]), 'pred': np.array([0, 0, 255]), 'inp': np.array([0, 255, 0])}
    #pointColorMap = ColorPalette(20).getColorMap()
    images = ((images.transpose((0, 2, 3, 1)) + 0.5) * 255).astype(np.uint8)
    for batchIndex in range(len(images)):
        image = images[batchIndex].copy()
        filename = options.test_dir + '/' + prefix +str(indexOffset + batchIndex) + '_image.png'
        cv2.imwrite(filename, image)
        for name, result_dict in dicts:
            for info in ['corner', 'icon', 'room']:
                cv2.imwrite(filename.replace('image', prefix + info + '_' + name), drawSegmentationImage(result_dict[info][batchIndex], blackIndex=0, blackThreshold=0.5))
                continue
            continue
        continue
    return

if __name__ == '__main__':
    args = parse_args()
    
    args.keyname = 'floorplan'
    #args.keyname += '_' + args.dataset

    if args.suffix != '':
        args.keyname += '_' + args.suffix
        pass
    
    args.checkpoint_dir = 'checkpoint/' + args.keyname
    args.test_dir = 'test/' + args.keyname

    print('keyname=%s task=%s started'%(args.keyname, args.task))

    main(args)
