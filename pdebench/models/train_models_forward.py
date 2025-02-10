

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="config", config_name="config_rdb")
def main(cfg: DictConfig):
    if cfg.args.model_name == "FNO":
        from pdebench.models.fno.train import run_training as run_training_FNO

        logger.info("FNO")
        run_training_FNO(
            if_training=cfg.args.if_training,
            continue_training=cfg.args.continue_training,
            num_workers=cfg.args.num_workers,
            modes=cfg.args.modes,
            width=cfg.args.width,
            initial_step=cfg.args.initial_step,
            t_train=cfg.args.t_train,
            training_type=cfg.args.training_type,
            num_channels=cfg.args.num_channels,
            batch_size=cfg.args.batch_size,
            epochs=cfg.args.epochs,
            learning_rate=cfg.args.learning_rate,
            scheduler_step=cfg.args.scheduler_step,
            scheduler_gamma=cfg.args.scheduler_gamma,
            model_update=cfg.args.model_update,
            flnm=cfg.args.filename,
            single_file=cfg.args.single_file,
            base_path=cfg.args.data_path,
            reduced_resolution=cfg.args.reduced_resolution,
            reduced_resolution_t=cfg.args.reduced_resolution_t,
            reduced_batch=cfg.args.reduced_batch,
            plot=cfg.args.plot,
            channel_plot=cfg.args.channel_plot,
            x_min=cfg.args.x_min,
            x_max=cfg.args.x_max,
            y_min=cfg.args.y_min,
            y_max=cfg.args.y_max,
            t_min=cfg.args.t_min,
            t_max=cfg.args.t_max,
        )
    elif cfg.args.model_name == "Unet":
        from pdebench.models.unet.train import run_training as run_training_Unet

        logger.info("Unet")
        run_training_Unet(
            if_training=cfg.args.if_training,
            continue_training=cfg.args.continue_training,
            num_workers=cfg.args.num_workers,
            initial_step=cfg.args.initial_step,
            t_train=cfg.args.t_train,
            in_channels=cfg.args.in_channels,
            out_channels=cfg.args.out_channels,
            epochs=cfg.args.epochs,
            learning_rate=cfg.args.learning_rate,
            batch_size=cfg.args.batch_size,
            unroll_step=cfg.args.unroll_step,
            ar_mode=cfg.args.ar_mode,
            pushforward=cfg.args.pushforward,
            scheduler_step=cfg.args.scheduler_step,
            scheduler_gamma=cfg.args.scheduler_gamma,
            model_update=cfg.args.model_update,
            flnm=cfg.args.filename,
            single_file=cfg.args.single_file,
            base_path=cfg.args.data_path,
            reduced_resolution=cfg.args.reduced_resolution,
            reduced_resolution_t=cfg.args.reduced_resolution_t,
            reduced_batch=cfg.args.reduced_batch,
            plot=cfg.args.plot,
            channel_plot=cfg.args.channel_plot,
            x_min=cfg.args.x_min,
            x_max=cfg.args.x_max,
            y_min=cfg.args.y_min,
            y_max=cfg.args.y_max,
            t_min=cfg.args.t_min,
            t_max=cfg.args.t_max,
        )
    elif cfg.args.model_name == "PINN":
        # not importing globally as DeepXDE changes some global PyTorch settings
        from pdebench.models.pinn.train import run_training as run_training_PINN

        logger.info("PINN")
        run_training_PINN(
            scenario=cfg.args.scenario,
            epochs=cfg.args.epochs,
            learning_rate=cfg.args.learning_rate,
            model_update=cfg.args.model_update,
            flnm=cfg.args.filename,
            seed=cfg.args.seed,
            input_ch=cfg.args.input_ch,
            output_ch=cfg.args.output_ch,
            root_path=cfg.args.root_path,
            val_num=cfg.args.val_num,
            if_periodic_bc=cfg.args.if_periodic_bc,
            aux_params=cfg.args.aux_params,
        )


if __name__ == "__main__":
    main()
