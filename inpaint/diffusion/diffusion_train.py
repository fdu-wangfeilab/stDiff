import sys
sys.path.append('./')

from Extenrnal.sc_DM.model.diffusion_train import *
from Extenrnal.sc_DM.inpaint.diffusion.diffusion_scheduler import RepaintNoiseScheduler

def normal_train(model, 
                 parameters: dict, 
                 max_iteration = 30000,
                 pred_type: str = 'noise',
                 data_type:str='atac',
                 batch_size = 1024,
                 diffusion_step=1000,
                 dataloader = None,
                 device = torch.device('cuda:0'),
                 is_nonzero_msk:bool=False,
                 is_class_condi=False,
                 is_tqdm=True,
                 is_tune=False,
                 condi_drop_rate=0.):


    noise_scheduler = RepaintNoiseScheduler(
        num_timesteps=diffusion_step,
        beta_schedule='cosine'
    )

    criterion = nn.MSELoss()
    model.to(device)
    num_epoch = max_iteration // len(dataloader)
    # 可学习 time_emb condi_emb 最佳参数 parameters={'lr': 0.098213970635869, 'momentum': 0.8892794756407683}
    # 可学习 time_emb 不加condition parameters={'lr': 0.09525346027279058, 'momentum': 0.582878382959882}

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=parameters['lr'],
        momentum= parameters['momentum']
    )


    if  is_tqdm:
        t_epoch = tqdm(range(num_epoch), ncols=100)
    else:
        t_epoch = range(num_epoch)

    model.train()

    for epoch in t_epoch:
        epoch_loss = 0.
        for i, (x, cell_type) in enumerate(dataloader):
            x,cell_type = x.to(device), cell_type.to(device).long()
            noise = torch.randn(x.shape).to(device)
            # timesteps = torch.randint_like(cell_type,1,diffusion_step).long().to(device)
            # 修改为
            timesteps = torch.randint(1, diffusion_step, (cell_type.shape[0],)).long().to(device)
            x_t = noise_scheduler.add_noise(x, noise, timesteps=timesteps)


            if not is_class_condi:
                cell_type = None
            if pred_type == 'noise':
                condi_preserve_flag = torch.rand(1) >= condi_drop_rate
                noise_pred = model(x_t, timesteps, cell_type, condi_flag=condi_preserve_flag)
                loss = criterion(noise_pred, noise)
            elif pred_type == 'x_start':
                x_0_pred = model(x_t, timesteps, cell_type)
                loss = criterion(x_0_pred, x)
            else:
                raise NotImplementedError

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0) # type: ignore
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/(i+1) # type: ignore
        if is_tqdm:
            t_epoch.set_postfix_str(f'{pred_type} loss:{epoch_loss:.5f}') # type: ignore
        if is_tune:
            session.report({'loss': epoch_loss})


def normal_train_no_guidance(model,
                 parameters: dict,
                 max_iteration=30000,
                 pred_type: str = 'noise',
                 data_type: str = 'atac',
                 batch_size=1024,
                 diffusion_step=1000,
                 dataloader=None,
                 device=torch.device('cuda:0'),
                 is_nonzero_msk: bool = False,
                 is_class_condi=False,
                 is_tqdm=True,
                 is_tune=False,
                 condi_drop_rate=0.):
    noise_scheduler = RepaintNoiseScheduler(
        num_timesteps=diffusion_step,
        beta_schedule='cosine'
    )

    criterion = nn.MSELoss()
    model.to(device)
    num_epoch = max_iteration // len(dataloader)
    # 可学习 time_emb condi_emb 最佳参数 parameters={'lr': 0.098213970635869, 'momentum': 0.8892794756407683}
    # 可学习 time_emb 不加condition parameters={'lr': 0.09525346027279058, 'momentum': 0.582878382959882}

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=parameters['lr'],
        momentum=parameters['momentum']
    )

    if is_tqdm:
        t_epoch = tqdm(range(num_epoch), ncols=100)
    else:
        t_epoch = range(num_epoch)

    model.train()

    for epoch in t_epoch:
        epoch_loss = 0.
        for i, x in enumerate(dataloader):
            # print(x.shape)
            x = torch.stack(x, dim=1).flatten(1)
            x = x.to(device)
            noise = torch.randn(x.shape).to(device)
            # timesteps = torch.randint_like(cell_type,1,diffusion_step).long().to(device)
            # 修改为
            timesteps = torch.randint(1, diffusion_step, (x.shape[0],)).long().to(device)
            x_t = noise_scheduler.add_noise(x, noise, timesteps=timesteps)


            if not is_class_condi:
                cell_type = None

            if pred_type == 'noise':
                condi_preserve_flag = torch.rand(1) >= condi_drop_rate
                noise_pred = model(x_t, timesteps, condi_flag=condi_preserve_flag)
                loss = criterion(noise_pred, noise)
            elif pred_type == 'x_start':
                x_0_pred = model(x_t, timesteps)
                loss = criterion(x_0_pred, x)
            else:
                raise NotImplementedError

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # type: ignore
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / (i + 1)  # type: ignore
        if is_tqdm:
            t_epoch.set_postfix_str(f'{pred_type} loss:{epoch_loss:.5f}')  # type: ignore
        if is_tune:
            session.report({'loss': epoch_loss})

def normal_train_palette(model,
                 parameters: dict,
                 max_iteration=30000,
                 pred_type: str = 'noise',
                 data_type: str = 'atac',
                 batch_size=1024,
                 diffusion_step=1000,
                 dataloader=None,
                 device=torch.device('cuda:0'),
                 is_nonzero_msk: bool = False,
                 is_class_condi=False,
                 is_tqdm=True,
                 is_tune=False,
                 condi_drop_rate=0.,
                 mask = None):
    noise_scheduler = RepaintNoiseScheduler(
        num_timesteps=diffusion_step,
        beta_schedule='cosine'
    )

    criterion = nn.MSELoss()
    model.to(device)
    num_epoch = max_iteration // len(dataloader)
    # 可学习 time_emb condi_emb 最佳参数 parameters={'lr': 0.098213970635869, 'momentum': 0.8892794756407683}
    # 可学习 time_emb 不加condition parameters={'lr': 0.09525346027279058, 'momentum': 0.582878382959882}

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=parameters['lr'],
        momentum=parameters['momentum']
    )

    if is_tqdm:
        t_epoch = tqdm(range(num_epoch), ncols=100)
    else:
        t_epoch = range(num_epoch)

    model.train()

    for epoch in t_epoch:
        epoch_loss = 0.
        for i, (x, x_cond) in enumerate(dataloader):
            # print(x.shape)
            # x = x.flatten(1)
            x = x.to(device)
            x_cond = x_cond.to(device)
            noise = torch.randn(x.shape).to(device)
            # timesteps = torch.randint_like(cell_type,1,diffusion_step).long().to(device)
            # 修改为
            timesteps = torch.randint(1, diffusion_step, (x.shape[0],)).long().to(device)
            x_t = noise_scheduler.add_noise(x, noise, timesteps=timesteps)
            mask = torch.tensor(mask).to(device)
            x_noisy = x_t * (1 - mask) + x * mask



            if pred_type == 'noise':
                condi_preserve_flag = torch.rand(1) >= condi_drop_rate
                noise_pred = model(x_noisy, timesteps, x_cond, condi_flag=condi_preserve_flag)
                loss = criterion(noise_pred * (1 - mask), noise * (1 - mask))
            elif pred_type == 'x_start':
                x_0_pred = model(x_t, timesteps)
                loss = criterion(x_0_pred, x)
            else:
                raise NotImplementedError

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # type: ignore
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / (i + 1)  # type: ignore
        if is_tqdm:
            t_epoch.set_postfix_str(f'{pred_type} loss:{epoch_loss:.5f}')  # type: ignore
        if is_tune:
            session.report({'loss': epoch_loss})





if __name__ =='__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["RAY_SESSION_DIR"] = "/home/lijiahao/ray_session"
    # ray_tune()
    train_scripts(model_name='scale2_self2_1024_latent.pt',
                  device = torch.device('cuda:0'),
                  pred_type='noise',
                  max_iteration=5000)
    # model = TransformerEncoder(
    #     input_dim=2000,
    #     num_classes=13,
    #     is_learned_timeebd=True,
    #     is_time_concat=True,
    #     is_condi_concat=True,
    #     num_heads=20,
    #     attn_types=['mlp','mlp']
    # )
    # device = torch.device('cuda:0')
    # model.to(device)
    # model.load_state_dict(torch.load('./model/ckpt/scale2_mlp2_1024.pt'))
    # print(model)