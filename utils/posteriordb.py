import numpy as np
import os
import json
import bridgestan as bs

        
class PDB():
    def __init__(self, model_n, modelname='', PDBPATH='/mnt/ceph/users/cmodi/PosteriorDB/'):
        self.model_path = os.path.join(PDBPATH, f'PDB_{model_n:02d}')
        self.id = f"{model_n:02d}"
        self.stan_model = self.code()
        self.data = self.dataset()
        self.get_reference_draws()
        if modelname == '':
            self.name = self.model_name()
        else:
            self.name = modelname
        # print("Model name :", self.name)
    
    def code(self):
        """
        Returns stan model code in model_path as a string
        """
        
        file_path = os.path.join(self.model_path, f'PDB_{self.id}.stan')
        with open(file_path) as f:
            contents = f.read()

        return contents

    
    def model_name(self):
        """
        Extract model name from model directory, if possible.
        Currently not implemented
        """
        return f"Model_{self.id}"
        # try:
        #     file_path = f'{self.model_path}/PDB_{self.id}.metadata.json'
        #     with open(file_path, "r") as f:
        #         contents = json.load(f)
        #     name = contents["name"]
        # except Exception as e:
        #     print("Exception occured in naming model\n", e)
        #     name = f"Model_{self.id}"
        # return name

    
    def dataset(self):
        """
        Reads and returns data from json format in model_path
        """
        file_path = os.path.join(self.model_path, f'PDB_{self.id}.data.json')
        with open(file_path, "r") as f:
            contents = json.load(f)

        return contents

    def get_reference_draws(self):
        """
        """
        
        params_file = os.path.join(self.model_path, f'PDB_{self.id}.samples.meta')
        with open(params_file, 'r') as f:
            self.parameters = [i for i in f.readline()]
            
        self.reference_draws = np.load(f'{self.model_path}/PDB_{self.id}.samples.npy')
        self.sample_chains = np.concatenate([i for i in self.reference_draws])
        self.samples = self.sample_chains.reshape(-1, self.sample_chains.shape[-1])

        
class BSDB(PDB):
    def __init__(self, model_n, PDBPATH):
        super().__init__(model_n, PDBPATH=PDBPATH)
        
        # Extract paths and load models
        prepend_path = os.path.join(self.model_path, f'PDB_{model_n:02d}')
        stanpath = prepend_path + '.stan'
        datapath = prepend_path + '.data.json'
        self.bsmodel = bs.StanModel.from_stan_file(stanpath, datapath)
        self.dimensions = self.bsmodel.param_unc_num()
        self.samples_unc = self.unconstrain(self.samples)

    def log_density(self, x):
        '''
        Wrapper to bridgestan log_density function to handle batches correctly
        '''
        if len(x.shape) == 1:
            return self.bsmodel.log_density(x)        
        elif len(x.shape) == 2:
            lps = np.array(list(map(self.bsmodel.log_density, x)))
            return lps

    def log_density_gradient(self, x):
        '''
        Wrapper to bridgestan log_density_gradient function to handle batches correctly
        '''
        if len(x.shape) == 1:
            return self.bsmodel.log_density_gradient(x)
            # try:
            #     return self.bsmodel.log_density_gradient(x)        
            # except Exception as e:
            #     print(e)
            #     return (None, None)
        elif len(x.shape) == 2:
            lp_gs = list(map(self.bsmodel.log_density_gradient, x))
            lps = np.array([i[0] for i in lp_gs])
            lpgs = np.array([i[1] for i in lp_gs])
            return (lps, lpgs)

    def unconstrain(self, x):
        '''
        Wrapper to bridgestan log_density_gradient function to handle batches correctly
        '''
        if len(x.shape) == 1:
            return self.bsmodel.param_unconstrain(x)        
        elif len(x.shape) == 2:
            samples = np.array(list(map(self.bsmodel.param_unconstrain, x)))
            return samples

    def constrain(self, x):
        '''
        Wrapper to bridgestan log_density_gradient function to handle batches correctly
        '''
        if len(x.shape) == 1:
            return self.bsmodel.param_constrain(x)        
        elif len(x.shape) == 2:
            samples = np.array(list(map(self.bsmodel.param_constrain, x)))
            return samples
        
    def dims(self):
        return self.dimensions


if __name__ == "__main__":

    failed = []
    for model_n in range(101):
        print()
        print(model_n)
        try:
            model = BSDB(model_n, PDBPATH='.')
            print('Dimensions : ', model.dims())
        except Exception as e:
            print('Exception occured in compiling : ', e)
            failed.append(model_n)
        break

    print('\nModels failed\n')
    print(failed)