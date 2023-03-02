import numpy as np
import torch
import glob


class Processor():
    def __init__(self, R_max=50, feh_min=-0.3, ofe_min=-4):
        self.R_max = R_max
        self.feh_min = feh_min
        self.ofe_min = ofe_min
    
    def preprocess(self, data_path):
        #y,y,z vx,vy,vz metals feh,ofe [mass] age/Gyr
        data = np.load(data_path)
        data = data.T

        #Exclude mass
        data = data[:,torch.tensor([True]*9+[False]+[True])]

        #Slice if wanted
        data_p = data
        data_p = torch.from_numpy(data_p).type(torch.float)

        #Constrain feh ofe and radius
        is_valid = (data_p[:,7] >=self.ofe_min)&(data_p[:,8]>=self.feh_min)&(torch.sqrt(torch.sum(data_p[:,:3]**2, dim=1))<=self.R_max)

        data_p = data_p[is_valid]

        #Scale data to train model
        self.mu = data_p.mean(dim=0)
        self.std = data_p.std(dim=0)
        data_p -= self.mu
        data_p /= self.std
        return data_p

    def topolar(self, data_car):
        rho = torch.sqrt(torch.sum(data_car[:,:2]**2, dim=1))
        phi = torch.arctan2(data_car[:,0], data_car[:,1])

        data_polar = torch.clone(data_car)

        data_polar[:,0] = rho
        data_polar[:,1] = phi
        return data_polar

    def frompolar(self, data_pol):
        is_valid = (data_pol[:,0]>=0)&(data_pol[:,1]<=np.pi)&(data_pol[:,1]>=-np.pi)

        data_car = torch.clone(data_pol[is_valid])

        x_ = data_car[:,0]*torch.cos(data_car[:,1])
        y_ = data_car[:,0]*torch.sin(data_car[:,1])

        

        data_car[:,0] = x_
        data_car[:,1] = y_

        return data_car

    def postprocessing(self, notprocessed):
        processed = notprocessed*self.std+self.mu
        return processed




class Processor_cond():
    def __init__(self, R_halfs=3, feh_min=-1.5, ofe_min=-5, N_min=0):
        self.R_halfs = R_halfs
        self.feh_min = feh_min
        self.ofe_min = ofe_min
        self.N_min = N_min

    
    #Some cleaning up and clarification could be done:
    #Use lists and append as output is list anyway, clean p and clrify what this does
    #Only data reading, and calculating N_stars, M_DM, M_stars_tot and M_tot, wich is already
    #included in the data.
    def get_data(self, folder):
        files = glob.glob(f"{folder}/*.npy")

        N_stars = np.array([np.load(name, mmap_mode="r").shape[1] for name in files])
        Cum_N_stars = np.cumsum(N_stars)

        #How many components?
        COMP_USE = 12
        Data = np.zeros((Cum_N_stars[-1], COMP_USE))
        M_stars_s = np.zeros(len(files))
        M_dm_s = np.zeros(len(files))

        for i,(file, start, end) in enumerate(zip(files, Cum_N_stars-N_stars, Cum_N_stars)):
            galaxy = np.load(file).T
            M_dm = float(file.split("_")[-1][:-4])
            M_dm_s[i] = M_dm

            M_stars = np.sum(galaxy[:,9])
            M_stars_s[i] = M_stars

            #Keep star masses to correct totel mass for constraining later
            #galaxy = galaxy[:,np.array([True]*9+[False]+1*[True])]
            galaxy = np.pad(galaxy, ((0,0),(0,1)), constant_values=M_stars+M_dm) #already here?

            Data[start:end] = galaxy
        return self.galaxysplit(Data, N_stars), N_stars, M_stars_s, M_dm_s


    #Work with Galaxy data and stack to distribution interpretation with diststack
    def galaxysplit(self, Data, N_stars):
        return np.split(Data,np.append(np.array([0]),np.cumsum(N_stars)))[1:-1]


    #Transform to 1 big array (interpretation as individual distribution points/samples) to feed flow
    def diststack(self, Data):
        return np.vstack(Data)


    #Data cleaning
    def constraindata(self, Data, info=True):
        Data_out = []
        N_stars = []
        M_stars = []
        N_old = 0
        for galaxy in Data:
            N_old += galaxy.shape[0]

            #Constrains on stars
            #No metallicity extreme stars
            is_valid = (galaxy[:,7] >=self.ofe_min)&(galaxy[:,8]>=self.feh_min)
            #Get half mass R, all stars equal mass, estimate: take the median R (as many stars inside as outside)
            R_halfmass = np.median(np.sqrt(np.sum(galaxy[:,:2]**2, axis=1)))
            #Exclude stars to far out
            is_valid = is_valid&(np.sqrt(np.sum(galaxy[:,:2]**2, axis=1))<=(self.R_halfs*R_halfmass))


            #Update Galaxy total mass by excluded stars, later update N stars
            galaxy[:, 11] -= np.sum(galaxy[~is_valid, 9])

            #Apply constrains now
            galaxy = galaxy[is_valid]
            N_star = galaxy.shape[0]

            #Constrain on galaxies (number of stars)
            if N_star>=self.N_min:
                Data_out.append(galaxy[:,np.array([True]*9+[False]+2*[True])]) # Exclude individual star masses, no longer needed
                N_stars.append(N_star)
                M_stars.append(np.sum(galaxy[:,9]))
        

        N_stars = np.array(N_stars)
        M_stars = np.array(M_stars)

        if info:
            print(f"Cut out {len(Data)-len(Data_out)} of {len(Data)} galaxies, {N_old-np.sum(N_stars)} of {N_old} stars (~{(N_old-np.sum(N_stars))/N_old*100 :.0f}%).")

        return Data_out, N_stars, M_stars

        


    def Data_to_flow(self, Data_c):
        Data_p = torch.from_numpy(np.copy(Data_c)).type(torch.float)

        ##DEBUG
        Data_p[:,-1] = torch.log10(Data_p[:,-1])

        self.mu = torch.from_numpy(Data_c.mean(axis=0)).type(torch.float)
        self.std = torch.from_numpy(Data_c.std(axis=0)).type(torch.float)
        Data_p -= self.mu
        Data_p /= self.std
        return Data_p


    def sample_to_Data(self, raw):
        Data = raw*self.std+self.mu
        return Data.numpy()

    def sample_Conditional(self, model, cond_indices, Condition):
        #Format: Contition is (N,n_cond) array cond_indices has length n_cond
        model.eval()
        with torch.inference_mode():
            res = (model.sample_Flow(Condition.shape[0], torch.from_numpy((Condition-(self.mu[cond_indices]))/(self.std[cond_indices])))).cpu() #Unsqueezed input

        return res