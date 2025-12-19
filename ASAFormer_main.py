from utils.data_loader import DataPrepare
from models.model import*
from utils.trainer import ModelTrainer
import os
from utils.evaluate_best_model import Evaluate_best_model
import torch

models_lib = {
    'ASAFormer':ASAFormer,
    }


for samples_set in [100,50]:
    class Args:
        # data参数设置
        label_dic = {'K0S0': 0, 'MS2': 1, 'K3S3': 2, 'K3S5': 3, 'MS4': 4, 'K3S7': 5, 'K5S3': 6, 'MS6': 7, 'K7S3': 8}
        len_window = 1024
        stride = 512
        type_standard = 'standard'  # 'min-max'
        use_val = True
        val_size = 0.1
        test_size = 0.3
        down_sampling = True  # False #
        # num samples(orignial)：{0: 13997, 1: 14129, 2: 13788, 3: 14526, 4: 14047, 5: 14100, 6: 14053, 7: 13975, 8: 14037}
        manual_down_samples =  [samples_set]*9
        add_noise = True
        snr = 5
        impulse_prob = 0.3
        impulse_amp = 0.05

        feature_dim = 3
        n_head = 1
        d_model = 128
        e_layers = 4

        #训练参数设置
        batch_size = 128
        epochs = 100 # #100
        learning_rate = 0.00001
        patience = 3
        num_classes = 9 #{'K0S0': 0, 'K3S3': 1, 'K3S5': 2, 'K3S7': 3, 'K5S3': 4, 'K7S3': 5, 'MS2': 6, 'MS4': 7, 'MS6': 8}
        iter_repeat = 1  #10 repeat
        raw_path = './'
        use_early_stop = True
        load_best_model = False  #True  #:load best model; False:train model

        optimizer = 'Adam' #'SGD'  #Adam?
        optim_params = {'lr': learning_rate, 'momentum': 0.9, 'weight_decay': 0.0002}
        scheduler_params = {'coslr': False, 'endlr': 0.0, 'gamma': 0.1, 'step_size': 30, 'warmup': True, 'lr_step': [120, 160], 'lr_factor': 0.01, 'warm_epoch': 5}

        cross_cond = True  #False  #False #
        test_condition =  '2R'  # 1R 2R 3R 10KN 20KN  30KN  (test condition)

        device = torch.device('cuda:0') ##'cpu', 'cuda:0'

    args = Args()

    models_params = {
        'ASAFormer': {'params': {'num_classes': args.num_classes, 'feature_dim': args.feature_dim,
                                'len_window': args.len_window, 'd_model': args.d_model,
                                'n_head': args.n_head, 'e_layers':args.e_layers},
                     'learning_rate': args.learning_rate,
                     'epochs': args.epochs}}


    raw_data_path = r'E:\【03】数据\TBM\主轴承'

    if args.cross_cond:
        results_path = args.raw_path + 'Cross_condition_results/'
        data_path = args.raw_path + 'Cross_condition_data/'
    else:
        results_path = args.raw_path + 'results/'
        data_path = args.raw_path + 'data/'

    for i_path in [results_path,data_path,results_path+'loss/',
                   results_path+'fig/',results_path+'checkpoints/',results_path+'csv/']:
        if not os.path.exists(i_path):
            os.makedirs(i_path)

    if not args.add_noise:
        args.snr = None
        args.impulse_prob = None
        args.noise_prob = None

    if not args.use_val:
        args.val_size = None

    # get_data
    data_prepare = DataPrepare(data_path,raw_data_path, args,)

    train_loader, val_loader, test_balanced_loader, test_imbalanced_loader = data_prepare.process()

    for model in models_lib.keys():
        args.model_name = model

        # iter的训练循环  (多次完整的train过程，检验模型稳定性)
        for i in range(args.iter_repeat):
            model = models_lib[args.model_name](**models_params[args.model_name]['params'])
            args.learning_rate = models_params[args.model_name]['learning_rate']
            args.epochs = models_params[args.model_name]['epochs']

            # 计算并打印参数数目
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total number of parameters: {total_params}")

            # 设置key information
            settings = '{}_{}_{}_lw{}_bs{}_ds{}'.format(args.cross_cond, args.test_condition,args.model_name,
                                                        args.len_window,args.batch_size,args.manual_down_samples,)

            ##直接调用best model，不再训练
            if args.load_best_model: #用均衡测试集跑
                best_model = Evaluate_best_model(model, args, test_balanced_loader, results_path)
                print('++++++++++++++++ Load best model for {} +++++++++++++'.format(settings))
                for i in range(args.iter_repeat):
                    best_model_pth_file = results_path + 'checkpoints/{}_best_model_iter{}.pth'.format(settings, i)
                    if os.path.exists(best_model_pth_file) and os.path.isfile(best_model_pth_file):
                        print(f'------iter: {i}-------')
                        best_model.evaluate_best_model(settings, best_model_pth_file, i, flag='balanced')
                    else:
                        print(f"No trained model for iter {i}")
                        break
                exit()

            ##训练模型
            trainer = ModelTrainer(model, args=args, results_path=results_path)
            print('==============================the {}-th train of {}==========================='.format(i,settings))
            trainer.train(train_loader,val_loader,test_balanced_loader, test_imbalanced_loader,settings,i)
            print('==============================the {}-th balanced test of {}==========================='.format(i,settings))
            trainer.test(test_balanced_loader,settings,i, flag='balanced')
            if test_imbalanced_loader is not None:
                print('==============================the {}-th imbalanced test of {}==========================='.format(i, settings))
                trainer.test(test_imbalanced_loader, settings, i, flag='imbalanced')
