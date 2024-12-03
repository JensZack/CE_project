from neuralop.training import Trainer
from neuralop.models import FNO
from neuralop.training import AdamW
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss
import numpy as np
import sys
from pathlib import Path
import torch
from vtk_parser import SimDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.animation as animation


device = "cpu"
SIM_DATASET = Path("/Users/zackjensen/data/DecayingTurbulence2D")
MODELS_FOLDER = Path("/Users/zackjensen/CE_project/models")


def train():

    train_data, test_data = SimDataset.train_test_datasets(SIM_DATASET)
    print(f"Test_idxs: {test_data.sim_idxs}")

    for training_resolution, epochs in zip([16, 32, 64], [2, 6, 10]):

        model = FNO(n_modes=(16, 16),
                    in_channels=2,
                    out_channels=2,
                    hidden_channels=32,
                    projection_channel_ratio=2)

        model = model.to(device)

        optimizer = AdamW(model.parameters(),
                          lr=8e-3,
                          weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

        trainer = Trainer(model=model, n_epochs=epochs,
                          device=device,
                          eval_interval=1,
                          verbose=True)

        l2loss = LpLoss(d=2, p=2)
        h1loss = H1Loss(d=2)

        train_loss = h1loss
        eval_losses = {'h1': h1loss, 'l2': l2loss}

        n_params = count_model_params(model)
        print(f'\nOur model has {n_params} parameters.')
        sys.stdout.flush()
        test_data.resolution = 32
        train_data.resolution = training_resolution

        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=16, shuffle=True)
        test_loaders = {32: test_loader}

        trainer.train(train_loader=train_loader,
                      test_loaders=test_loaders, # noqa
                      optimizer=optimizer,
                      scheduler=scheduler,
                      regularizer=False,
                      training_loss=train_loss,
                      eval_losses=eval_losses)

        model.save_checkpoint(MODELS_FOLDER, f"model_{training_resolution}_6")


def load_model(model_name):
    model = FNO.from_checkpoint(MODELS_FOLDER, model_name)
    return model


def velocity_to_vorticity(vf: torch.Tensor) -> np.ndarray:
    dx = 1 / (vf.shape[1] - 1)

    dvdx = torch.gradient(vf[0, ...], axis=1)[0] / dx  # noqa
    dudy = torch.gradient(vf[1, ...], axis=0)[0] / dx  # noqa
    return (dvdx - dudy).detach().numpy()


def vorticity_anim_2(vf_ref, model, res: int):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    idx = 1
    dt = .01
    vorticity_ref = velocity_to_vorticity(vf_ref[idx])
    velocity_pred = model(vf_ref[idx - 1].unsqueeze(0))
    vorticity_pred = velocity_to_vorticity(velocity_pred.squeeze(0))

    f.suptitle(f'Vorticity Model Training Resoluiton {res}')
    ax1.imshow(vorticity_ref, label='vorticity_reference')
    ax2.imshow(vorticity_pred, label='vorticity_prediction')
    ax1.set_title(f'Time: {(idx * dt):.4f} s')

    def update(frame):
        nonlocal idx
        nonlocal velocity_pred
        idx += 1
        # for each frame, update the data stored on each artist.
        if idx > 10:
            return (f,)

        ax1.set_title(f'Time: {(idx * dt):.4f} s')
        vorticity_ref_ = velocity_to_vorticity(vf_ref[idx])
        velocity_pred = model(velocity_pred)
        vorticity_pred = velocity_to_vorticity(velocity_pred.squeeze(0))
        ax1.imshow(vorticity_ref_, label='vorticity_reference')
        ax2.imshow(vorticity_pred, label='vorticity_prediction')
        ax1.set_xlabel('Vorticity Reference')
        ax2.set_xlabel('Vorticity Prediction')

        return (f,)

    ani = animation.FuncAnimation(fig=f, func=update, interval=500, repeat=True, frames=10)
    plt.show()


def plot_model(res, steps = 10):
    model = load_model(f"model_{res}_6")
    dataset = sim_dataset = SimDataset(SIM_DATASET)
    vf_ref = dataset.frames[4]
    v_pred = vf_ref[0]
    for _ in range(steps):
        v_pred = model(v_pred.unsqueeze(0)).squeeze(0)
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    dt = .01
    vorticity_ref = velocity_to_vorticity(vf_ref[steps])
    vorticity_pred = velocity_to_vorticity(v_pred.squeeze(0))

    f.suptitle(f'Vorticity Model Training Resolution {res}')
    ax1.imshow(vorticity_ref, label='vorticity_reference')
    ax2.imshow(vorticity_pred, label='vorticity_prediction')
    ax1.set_title(f'Time: {(steps * dt):.4f} s')
    ax1.set_xlabel('Vorticity Reference')
    ax2.set_xlabel('Vorticity Prediction')
    plt.show()


def main():
    pass

if __name__ == "__main__":
    main()
