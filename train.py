import os.path
import math
import argparse
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
import torch

from utils import logger as util_logger
from utils import image as image_util
from utils import options as option

from utils.dataset import UPMDataset
from models.ludvae import LUDVI

def main(json_path: str ='./train_sidd.jsonc'):

    """
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    image_util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter, init_path = option.find_last_checkpoint(opt['path']['models'])
    opt['path']['pretrained_net'] = init_path
    current_step = init_iter

    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'train'
    util_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = UPMDataset(dataset_opt)

            labels = np.asarray(train_set.labels)
            unique, counts = np.unique(labels, return_counts=True)
            logger.info('balancing classes', dict(zip(unique,counts)))
            weights = np.zeros_like(labels).astype(float)
            for class_id, count in zip(unique, counts):
                weights[labels==class_id] = len(weights)/count
                print('class:', class_id, 'count:', count, 'total:', len(weights), 'weight:', len(weights)/count, 'real', weights[labels==class_id].mean())
            train_sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            print('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            train_loader = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      shuffle=dataset_opt['dataloader_shuffle'],
                                      num_workers=dataset_opt['dataloader_num_workers'],
                                      sampler=train_sampler,
                                      drop_last=True,
                                      pin_memory=True)
        elif phase == 'test':
            test_set = UPMDataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = LUDVI(opt)

    logger.info(model.info_network())
    model.init_train()
    logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    # [PRINT] detectar GPU/CPU e mostrar qual device será usado (CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GPU] Device selecionado: {device}")

    num_epochs = 2  # você pode trocar por opt['train']['n_epochs'], se existir
    for epoch in range(num_epochs):  # loop de épocas
        print(f"Iniciando época {epoch+1}/{num_epochs}")

        for i, train_data in enumerate(train_loader):

            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # [PRINT] Mostrar informações do batch (tamanhos das tensões)
            # -------------------------------
            batch_sizes = {k: v.shape if isinstance(v, torch.Tensor) else None for k, v in train_data.items()}
            print(f"[Batch] Epoch {epoch+1}, Iter {current_step}, tamanhos: {batch_sizes}")

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # --------------------------
            # 4) training information
            # --------------------------
            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = model.current_log()  # such as loss
                message = (
                    f"<epoch:{epoch+1:3d}, iter:{current_step:8,d}, "
                    f"lr:{model.current_learning_rate():.3e}> "
                )
                for k, v in logs.items():  # merge log information into message
                    message += f"{k}: {v:.3e} "
                logger.info(message)
                print(f"[Log] {message}")

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0:
                logger.info('Saving the model.')
                print(f"[Checkpoint] Salvando modelo no passo {current_step}...")
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0:
                idx = 0

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    image_util.mkdir(img_dir)

                    # -------------------------------
                    # [PRINT] Configuração do caminho do arquivo salvo
                    # -------------------------------
                    print(f"[Teste] Salvando resultado de {img_name} (label={{label}}) no passo {current_step} em {img_dir}")

                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    img = image_util.tensor2uint(visuals['img'])
                    img_t = image_util.tensor2uint(visuals['img_t'])
                    label = int(visuals['label'])

                    # -----------------------
                    # save
                    # -----------------------
                    save_img_path = os.path.join(
                        img_dir, f'{img_name}_{label}.png'
                    )
                    save_img_t_path = os.path.join(
                        img_dir, f'{img_name}_{label}_to_{1-label}_{current_step}.png'
                    )

                    image_util.imsave(img_t, save_img_t_path)

                    if current_step == opt['train']['checkpoint_test']:
                        image_util.imsave(img, save_img_path)

    logger.info('Saving the final model.')
    print("[Final] Salvando modelo final…")
    model.save('latest')
    logger.info('End of training.')
    print("Geração concluída com sucesso!")

if __name__ == '__main__':
    main()
