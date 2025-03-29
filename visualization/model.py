from dataset import *
import os
import datetime
import re
from pathlib import Path
import argparse
import npfl138
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
npfl138.require_version("2425.5")
torch.set_default_device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=1e-2, type=float, help="Model learning rate.")
parser.add_argument("--learning_rate_final", default=1e-3, type=float, help="Final model learning rate.")
parser.add_argument("--patience", default=5, type=int, help="Early stopping patience.")
parser.add_argument("--hidden_dims", default=[64, 64], type=list[int], help="Dimensions of hidden layers.")
parser.add_argument("--weight_decay", default=1e-2, type=float, help="Weight decay.")
parser.add_argument("--evaluate", default=False, action="store_true", help="Perform just an evaluation.")


DATA_PATH = Path("/mount/data/preprocessed_dataset")
TRAIN_PATH = DATA_PATH / "train"
VAL_PATH = DATA_PATH / "val"
MODEL_PATH = Path("./model/model.pt")
CLASSES2EVAL = [10, 30, 40] # Only evaluate on these classes

class EarlyStopper:
    def __init__(self, patience: int, best_model_filename: str | None) -> None:
        self._patience = patience
        self._best_loss = np.inf
        self._epochs_without_improvement = 0
        self._best_model_filename = best_model_filename

    def __call__(self, model, epoch, logs):
        current_loss = logs["dev_loss"]
        if current_loss < self._best_loss:
            self._best_loss = current_loss
            self._epochs_without_improvement = 0

            if self._best_model_filename is not None:
                model.save_weights(self._best_model_filename)
        else:
            self._epochs_without_improvement += 1
            if self._epochs_without_improvement >= self._patience:
                return npfl138.TrainableModule.STOP_TRAINING


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, height, width):
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)
            
        if isinstance(self.hidden_dim, int):
            self.hidden_dim = [self.hidden_dim] * self.num_layers
        
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
            cell_list.append(ConvLSTMCell(cur_input_dim, self.hidden_dim[i], self.kernel_size, self.bias))
        
        self.cell_list = nn.ModuleList(cell_list)
    
    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
            
        b, t, c, h, w = input_tensor.size()
        
        if hidden_state is None:
            hidden_state = self._init_hidden(b, h, w)
            
        layer_output_list = []
        last_state_list = []
        
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], (h, c))
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
            
        return layer_output_list, last_state_list
    
    def _init_hidden(self, batch_size, height, width):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, height, width))
        return init_states

class DummyModel(npfl138.TrainableModule):
    def __init__(self, input_channels):
        super().__init__()
        
        self.convlstm = ConvLSTM(
            input_dim=input_channels,
            hidden_dim=[64, 64],
            kernel_size=3,
            num_layers=2,
            batch_first=True,
            return_all_layers=False
        )
        
        # Final prediction layer
        self.conv_output = nn.Conv2d(
            in_channels=64,
            out_channels=input_channels,
            kernel_size=3,
            padding=1
        )
    
    def forward(self, x):
        # x shape: [batch_size, seq_len=5, channels, height, width]
        
        # Pass through ConvLSTM
        layer_output_list, _ = self.convlstm(x)
        
        # Get the output of the last layer's last time step
        last_layer_output = layer_output_list[-1][:, -1]
        
        # Generate the prediction
        prediction = self.conv_output(last_layer_output)
        
        return prediction
    
class SeqSimpleDataset(SeqGreenEarthNetDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        dict_data = super().__getitem__(idx)
        inputs = dict_data["inputs"].astype(np.float32)
        inputs = np.nan_to_num(inputs, nan=0.0)
        targets = dict_data["targets"].astype(np.float32)
        targets = np.nan_to_num(targets, nan=0.0)
        return (inputs, targets)
    

def _compute_evi(predictions):
    # input_channels=["red", "green", "blue", "nir"]
    # EVI = 2.5 * ((NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1))
    #print(predictions.shape)
    red = predictions[:, 0]
    green = predictions[:, 1]
    blue = predictions[:, 2]
    nir = predictions[:, 3]

    return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)


class MaskedMSELoss():
    def __init__(self, classes):
        self._classes = classes

    def __call__(self, pred, target):
        target_evi = target[:, 0, 0]
        class_mask = target[:, 0, 1]
        target_clr = target[:, :, 2:]

        valid_mask = (torch.isin(class_mask, torch.tensor(self._classes))) & (~torch.isnan(target_evi)) & (target_evi >= -1) & (target_evi <= 1)
        valid_mask = valid_mask.unsqueeze(1)
        pred = pred.unsqueeze(1)
        valid_mask = valid_mask.unsqueeze(2).expand(-1, -1, 4, -1, -1)
        target_clr = target_clr[valid_mask]
        pred = pred[valid_mask]
        assert pred.shape == target_clr.shape
        if len(pred) == 0:
            return torch.tensor(0.0, requires_grad=True)
        return torch.nn.functional.mse_loss(pred, target_clr)



def main(args: argparse.Namespace):
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)


    args.logdir = os.path.join("model/logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))



    info_list = list(ADDITIONAL_INFO_DICT.keys())
    input_channels=["red", "green", "blue", "nir"]
    target_channels=["evi", "class", "red", "green", "blue", "nir"]
    ds_train = SeqSimpleDataset(
        folder=TRAIN_PATH,
        input_channels=input_channels,
        target_channels=target_channels,
        additional_info_list=info_list,
        time=True,
        use_mask=True,
    )
    ds_val = SeqSimpleDataset(
        folder=VAL_PATH,
        input_channels=input_channels,
        target_channels=target_channels,
        additional_info_list=info_list,
        time=True,
        use_mask=True,
    )


    dl_train = DataLoader(ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        #collate_fn=custom_collate_fn,
        generator=torch.Generator(device=device),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=True,
        #collate_fn=custom_collate_fn,
        generator=torch.Generator(device=device),
    )

    model = DummyModel(args, len(input_channels))

    # TODO
    if args.evaluate:
        model.load_weights(MODEL_PATH)
        predictions = model.predict(dl_val, data_with_labels=True)
        for i, prediction in enumerate(predictions):
            prediction = np.transpose(prediction[:3, :, :], (1, 2, 0))
            min_val = prediction.min()
            max_val = prediction.max()
            prediction = (prediction - min_val) / (max_val - min_val)
            image_np = (prediction * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            pil_image.save(f'model/logs/prediction_{i}.png')
        return



    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(dl_train) * args.epochs, eta_min=args.learning_rate_final
    )
    model.configure(
        optimizer=optimizer,
        scheduler=scheduler,
        loss=MaskedMSELoss(CLASSES2EVAL),
        logdir=args.logdir,
    )

    model.fit(
        dl_train,
        dev=dl_val,
        epochs=args.epochs,
        log_graph=True,
        callbacks=[EarlyStopper(args.patience, MODEL_PATH)],
    )


    os.makedirs(args.logdir, exist_ok=True)


    model.save_weights(MODEL_PATH)

    #dl_test = dl_val

    #output = model.predict(dl_val, data_with_labels=True)
    #with open(
    #    os.path.join(args.logdir, "predictions.txt"), "w", encoding="utf-8"
    #) as predictions_file:
    #    # Perform the prediction on the test data. The line below assumes you have
    #    # a dataloader `test` where the individual examples are `(image, target)` pairs.
    #    for prediction in model.predict(dl_test, data_with_labels=True):
    #        print(np.argmax(prediction), file=predictions_file)





if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)