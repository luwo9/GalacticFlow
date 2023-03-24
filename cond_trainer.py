import numpy as np
import torch
import flowcode
import time


train_data = torch.load("cond_trainer/data_cond_trainer.pth")
model = torch.load("cond_trainer/model_cond_trainer.pth").to("cuda:5")
train_parameters = np.load("cond_trainer/params_cond_trainer.npy")
filename = str(np.load("cond_trainer/filename_cond_trainer.npy"))

train_loss_saver = []

start = time.perf_counter()

flowcode.train_flow(model, train_data, train_parameters[:-4].astype("int"), int(train_parameters[-4]), lr=train_parameters[-3], batch_size=int(train_parameters[-2]), loss_saver=train_loss_saver, gamma=train_parameters[-1], give_textfile_info=True)

end = time.perf_counter()

torch.save(model.state_dict(), f"saves/{filename}.pth")
np.save(f"saves/loss_{filename}.npy",np.array(train_loss_saver+[end-start]))
