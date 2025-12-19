import torch.optim as optim
from .Warmupscheduler import WarmupMultiStepLR

def Initial_optimizer(opti_settings, args):

    if(args.optimizer == 'Adam'):
        print('=====> Using Adam optimizer')
        model_optimizer = optim.Adam([opti_settings])
    else:
        print('=====> Using SGD optimizer')
        model_optimizer = optim.SGD([opti_settings])

    return model_optimizer

def Initial_scheduler(model_optimizer, sche_settings, args):

    if(sche_settings['coslr']):
        print("===> Module model : Using coslr eta_min={}".format(sche_settings['endlr']))
        model_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            model_optimizer, args.epochs, eta_min=sche_settings['endlr'])
    elif sche_settings['warmup']:  # 带有热身阶段的多步学习策略
        print("===> Module model : Using warmup")
        model_scheduler = WarmupMultiStepLR(model_optimizer, sche_settings['lr_step'],  # lr_step学习率下降的epoch
                                                           gamma=sche_settings['lr_factor'],
                                                           warmup_epochs=sche_settings[
                                                               'warm_epoch'])  # lr_factor是学习率衰减因子，warm_epoch学习率上升达到预定学习率的epoch
    else:  # 步骤学习策略，学习率每隔epochs衰减一次
        model_scheduler = optim.lr_scheduler.StepLR(model_optimizer,
                                                                   step_size=sche_settings['step_size'],
                                                                   gamma=sche_settings['gamma'])

    return model_scheduler