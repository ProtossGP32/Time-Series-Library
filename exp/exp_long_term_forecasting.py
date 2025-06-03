from __future__ import annotations

import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from torch import optim
from utils.augmentation import run_augmentation
from utils.augmentation import run_augmentation_single
from utils.dtw_metric import accelerated_dtw
from utils.dtw_metric import dtw
from utils.metrics import metric
from utils.tools import adjust_learning_rate
from utils.tools import EarlyStopping
from utils.tools import visual

warnings.filterwarnings("ignore")


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, setting, encoder=None, scaler=None):
        data_set, data_loader = data_provider(
            self.args, flag, setting, encoder, scaler
        )
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate
        )
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len :, :]
                ).float()
                dec_inp = (
                    torch.cat(
                        [batch_y[:, : self.args.label_len, :], dec_inp], dim=1
                    )
                    .float()
                    .to(self.device)
                )
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                else:
                    outputs = self.model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark
                    )
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(
                    self.device
                )

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(
            flag="train", setting=setting
        )
        vali_data, vali_loader = self._get_data(flag="val", setting=setting)
        test_data, test_loader = self._get_data(flag="test", setting=setting)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        total_time_train = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len :, :]
                ).float()
                dec_inp = (
                    torch.cat(
                        [batch_y[:, : self.args.label_len, :], dec_inp], dim=1
                    )
                    .float()
                    .to(self.device)
                )

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )

                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(
                            self.device
                        )
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark
                    )

                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(
                        self.device
                    )
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print(
                "Epoch: {} cost time: {}".format(
                    epoch + 1, time.time() - epoch_time
                )
            )
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
        end_time = time.time()
        total_training_time = end_time - total_time_train
        hours, rem = divmod(total_training_time, 3600)
        minutes, seconds = divmod(rem, 60)
        time_str = "{:0>2}:{:0>2}:{:05.2f}".format(
            int(hours), int(minutes), seconds
        )
        print(f"Total training time:{time_str}")

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test", setting=setting)
        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(
                    os.path.join("./checkpoints/" + setting, "checkpoint.pth")
                )
            )
        input_list = []
        preds = []
        trues = []
        folder_path = "./test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                # Add right after the dataloader provides the batch
                print("Original shapes from dataloader:")
                print("batch_x raw shape:", batch_x.shape)
                print("batch_y raw shape:", batch_y.shape)
                print("batch_x_mark raw shape:", batch_x_mark.shape)
                print("batch_y_mark raw shape:", batch_y_mark.shape)
                print("label_len:", self.args.label_len)
                print("pred_len:", self.args.pred_len)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len :, :]
                ).float()
                dec_inp = (
                    torch.cat(
                        [batch_y[:, : self.args.label_len, :], dec_inp], dim=1
                    )
                    .float()
                    .to(self.device)
                )
                # Add right after dec_inp is created
                print("After dec_inp creation:")
                print("dec_inp shape:", dec_inp.shape)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                else:
                    outputs = self.model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark
                    )

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, :]
                batch_y = batch_y[:, -self.args.pred_len :, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(
                            outputs,
                            [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])],
                        )
                    outputs = test_data.inverse_transform(
                        outputs.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)
                    batch_y = test_data.inverse_transform(
                        batch_y.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y
                input_list.append(batch_x.detach().cpu().numpy())

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(
                            input.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)
                    gt = np.concatenate(
                        (input[0, :, -1], true[0, :, -1]), axis=0
                    )
                    pd = np.concatenate(
                        (input[0, :, -1], pred[0, :, -1]), axis=0
                    )
                    visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        input_list = np.concatenate(input_list, axis=0)
        print("test shape:", preds.shape, trues.shape, input_list.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        input_list = input_list.reshape(-1, input_list.shape[-2], input_list.shape[-1])
        print("test shape:", preds.shape, trues.shape, input_list.shape)

        # result save
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = "Not calculated"

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print("mse:{}, mae:{}, dtw:{}".format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", "a")
        f.write(setting + "  \n")
        f.write("mse:{}, mae:{}, dtw:{}".format(mse, mae, dtw))
        f.write("\n")
        f.write("\n")
        f.close()

        np.save(
            folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe])
        )
        np.save(folder_path + "pred.npy", preds)
        np.save(folder_path + "true.npy", trues)
        np.save(folder_path + "input.npy", input_list)

        return

    def predict(self, setting, load=True):
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + "/" + "checkpoint.pth"
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        pred_data, pred_loader = self._get_data(flag="pred", setting=setting)
        self.model.eval()
        preds = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                # Create zeros tensor of the right size (not based on slicing batch_y)
                future_zeros = torch.zeros((batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]), device=batch_y.device)
                # Concatenate with available label timesteps
                dec_inp = torch.cat([batch_y, future_zeros], dim=1)
                # IMPORTANT: Pad batch_y_mark to match dec_inp's sequence length
                if batch_y_mark.shape[1] != dec_inp.shape[1]:
                    padding_needed = dec_inp.shape[1] - batch_y_mark.shape[1]
                    last_mark = batch_y_mark[:, -1:, :].repeat(1, padding_needed, 1)
                    batch_y_mark = torch.cat([batch_y_mark, last_mark], dim=1)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                else:
                    outputs = self.model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark
                    )

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, :]
                batch_y = batch_y[:, -self.args.pred_len :, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    actual_shape = (outputs.shape[0], outputs.shape[1], batch_y.shape[-1])
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(
                            outputs,
                            [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])],
                        )
                    outputs = pred_data.inverse_transform(
                        outputs.reshape(actual_shape[0] * actual_shape[1], -1)
                    ).reshape(actual_shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                preds.append(pred)
            preds = np.concatenate(preds, axis=0)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

            # # result save
            # folder_path = './workflow/results/'
            # if not os.path.exists(folder_path):
            #     os.makedirs(folder_path)

            # np.save(folder_path+'real_prediction.npy', preds)

        return preds
    
