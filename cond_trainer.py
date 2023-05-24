import numpy as np
import torch
import flowcode
import time
import sys

import device_use

file_ext = sys.argv[1]
train_data = torch.load("cond_trainer/data_cond_trainer.pth")
model = torch.load("cond_trainer/model_cond_trainer.pth")
train_parameters = np.load("cond_trainer/params_cond_trainer.npy")
filename = str(np.load("cond_trainer/filename_cond_trainer.npy"))

#Signal that loading is complete and the file can be overwritten safely
np.save("cond_trainer/loading_complete.npy", np.array([1]))

train_loss_saver = []

start = time.perf_counter()

flowcode.train_flow(model, train_data, train_parameters[:-4].astype("int"), int(train_parameters[-4]), lr=train_parameters[-3], batch_size=int(train_parameters[-2]), loss_saver=train_loss_saver, gamma=train_parameters[-1], give_textfile_info=file_ext)

end = time.perf_counter()

torch.save(model.state_dict(), f"saves/{filename}.pth")
np.save(f"saves/loss_{filename}.npy",np.array(train_loss_saver+[end-start]))
