import os
import torch
from model import LandSeg as net
import torch.backends.cudnn as cudnn
import DataSet as myDataLoader
from argparse import ArgumentParser
from utils import train, val, netParams, save_checkpoint, poly_lr_scheduler
import torch.optim.lr_scheduler
from loss import TotalLoss

# We would like to thank the authors of TwinLiteNet for their repository containing the different classes used in this work.
# Link to adopted work: https://github.com/chequanghuy/TwinLiteNet
def train_net(args):
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()

    model = net.LandSeg()

    if num_gpus > 1:
        model = torch.nn.DataParallel(model)

    args.savedir = args.savedir + '/'

    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)


    trainLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)


    valLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(valid=True),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if cuda_available:
        args.onGPU = True
        model = model.cuda()
        cudnn.benchmark = True

    criteria = TotalLoss()
    
    start_epoch = 0
    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr, (0.9,0.999),eps=1e-08, weight_decay=2e-5)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
           # start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            #model.load_state_dict(checkpoint)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}'"
                .format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    optimizer = torch.optim.Adam(model.parameters(), lr, (0.9,0.999),eps=1e-08, weight_decay=2e-5)
    
    for epoch in range(start_epoch, args.max_epochs):

        model_file_name = args.savedir + os.sep + 'model_{}.pth'.format(epoch)
        poly_lr_scheduler(args, optimizer, epoch)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Learning rate: " +  str(lr))

       
        model.train()
        train(args, trainLoader, model, criteria, optimizer, epoch)
        if (epoch%25==0):
            model.eval()
            # validation
            landingSurfaceResults = val(valLoader, model)
        
            msg =  'Floor Segment: Acc({seg_acc:.3f})    IOU ({seg_iou:.3f})    mIOU({seg_miou:.3f})'.format(
                          seg_acc=landingSurfaceResults[0],seg_iou=landingSurfaceResults[1],seg_miou=landingSurfaceResults[2])
            print(msg)
        
            torch.save(model.state_dict(), model_file_name)
        
        save_checkpoint({
            'epoch': epoch + 1,	
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': lr
        }, args.savedir + 'checkpoint.pth.tar')

        


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=50, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=2, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
    parser.add_argument('--step_loss', type=int, default=10, help='Decrease learning rate after how many epochs.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--savedir', default='', help='Directory to save the results')
    parser.add_argument('--resume', type=str, default='', help='Use this flag to load last checkpoint')
    parser.add_argument('--pretrained', default='', help='Pretrained LandSeg Weights.')

    train_net(parser.parse_args())

