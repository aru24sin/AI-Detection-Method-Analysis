import os
import time
import torch
from tensorboardX import SummaryWriter

from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions

def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    #val_opt.dataroot = os.path.join(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    
    #val_opt.jpg_method = 'pil'
    #val_opt.resize_or_crop = 'crop'
    #val_opt.loadSize = val_opt.cropSize
    #val_opt.batch_size = 32
    
    val_opt.jpg_method = ['pil']

    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]

    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]
    return val_opt

if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    val_opt = get_val_opt()

    data_loader = create_dataloader(opt)
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    
    model = Trainer(opt)
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)

    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
    
        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size
        
            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)
            if model.total_steps % opt.save_latest_freq == 0:
                print('saving the latest model %s (epoch %d, model.total_steps %d)' % (opt.name, epoch, model.total_steps))
                model.save_networks('latest')
            # print("Iter time: %d sec" % (time.time()-iter_data_time))
            # iter_data_time = time.time()
            if epoch % opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' % (epoch, model.total_steps))
                model.save_networks('latest')
                model.save_networks(epoch)
        
            # Validation
            model.eval()
            acc, ap = validate(model.model, val_opt)[:2]
            val_writer.add_scalar('accuracy', acc, model.total_steps)
            val_writer.add_scalar('ap', ap, model.total_steps)
            print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
       
            early_stopping(acc, model)
            if early_stopping.early_stop:
                cont_train = model.adjust_learning_rate()
                if cont_train:
                    print("Learning rate dropped by 10, continue training...")
                    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
                else:
                    print("Early stopping.")
                    break
            model.train()
       






#if __name__ == '__main__':
#    opt = TrainOptions().parse()
#    opt.dataroot = os.path.join(opt.dataroot, opt.train_split)
#   # dataloader, dataset_size = create_dataloader(opt)
    
#    train_dataloader = create_dataloader(opt)
    
#    dataset_size = len(train_dataloader)
#    print('#training images = %d' % dataset_size)

#    val_opt = get_val_opt()
   # val_dataloader, val_dataset_size = create_dataloader(val_opt)
#    val_dataloader = create_dataloader(val_opt)

#    train_logger = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, 'train'))
#    val_logger = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, 'val'))

#    early_stopping = EarlyStopping(opt)

#    trainer = Trainer(opt)
#    total_steps = 0

#    print("Start Training")

#    for epoch in range(opt.niter):
#        epoch_start_time = time.time()
#        epoch_iter = 0

#        for i, data in enumerate(train_dataloader):
#            iter_start_time = time.time()

#            total_steps += opt.batch_size
#            epoch_iter += opt.batch_size

#            trainer.set_input(data)
#            trainer.optimize_parameters()

#            if total_steps % opt.loss_freq == 0:
#                train_logger.add_scalar('loss', trainer.get_current_errors(), total_steps)

#            if total_steps % opt.save_latest_freq == 0:
#                print(f'saving the latest model (epoch {epoch}, total_steps {total_steps})')
#                print("saving the latest model epoch " + str(epoch) + ", totoal_steps " + str(total_steps))
#                trainer.save('latest')

#        if epoch % opt.save_epoch_freq == 0:
            #print(f'saving the model at the end of epoch {epoch}, iters {total_steps}')
#            print("saving the model at the end of epoch " + str(epoch) + ". iters " + str(total_stpes))
#            trainer.save('latest')
#            trainer.save(epoch)

#        val_loss = validate(val_opt, val_dataloader, trainer, epoch, val_logger)

#        early_stopping(val_loss, trainer)

        #print(f'End of epoch {epoch} / {opt.niter} \t Time Taken: {time.time() - epoch_start_time} sec')
#        print("Time Taken: " + str (time.time() - epoch_start_time) + ' sec')

#        if early_stopping.early_stop:
#            print("Early stopping")
#            break

#    train_logger.close()
#    val_logger.close()
