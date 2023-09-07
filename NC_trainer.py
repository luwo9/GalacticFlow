import numpy as np
import torch
import flowcode
import time
import sys

import device_use

file_ext = sys.argv[1]
train_data = torch.load("NC_trainer/data_NC_trainer.pth")
model = torch.load("NC_trainer/model_NC_trainer.pth").to(device_use.device_use)
train_parameters = np.load("NC_trainer/params_NC_trainer.npy")
filename = str(np.load("NC_trainer/filename_NC_trainer.npy"))

train_loss_saver = []

start = time.perf_counter()

flowcode.train_flow(model, train_data, [], int(train_parameters[-4]), lr=train_parameters[-3], batch_size=int(train_parameters[-2]), loss_saver=train_loss_saver, gamma=train_parameters[-1], give_textfile_info=file_ext)

end = time.perf_counter()

torch.save(model.state_dict(), f"saves/{filename}.pth")
np.save(f"saves/loss_{filename}.npy",np.array(train_loss_saver+[end-start]))
