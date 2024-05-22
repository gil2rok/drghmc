import numpy as np

class Sampler():

    def __init__(self):
        self.samples = []
        self.accepts = []
        self.Hs = []
        self.counts = []
        self.i = 0

    def to_array(self):
        for key in self.__dict__:
            if type(self.__dict__[key]) == list:
                self.__dict__[key] = np.array(self.__dict__[key])            

    def to_list(self):
        for key in self.__dict__:
            if type(self.__dict__[key]) == np.ndarray:
                self.__dict__[key] = list(self.__dict__[key])

    def appends(self, q, acc, Hs, count):
        self.i += 1
        self.accepts.append(acc)
        self.samples.append(q)
        self.Hs.append(Hs)
        self.counts.append(count)
        
    def save(self, path):
        pass



class DualAveragingStepSize():
    def __init__(self, initial_step_size, target_accept=0.65, gamma=0.05, t0=10.0, kappa=0.75, nadapt=0):
        self.initial_step_size = initial_step_size 
        self.mu = np.log(10 * initial_step_size)  # proposals are biased upwards to stay away from 0.
        self.target_accept = target_accept
        self.gamma = gamma
        self.t = t0
        self.kappa = kappa
        self.error_sum = 0
        self.log_averaged_step = 0
        self.nadapt = nadapt
        
    def update(self, p_accept):

        if np.isnan(p_accept) or np.isinf(p_accept): p_accept = 0.
        if p_accept > 1: p_accept = 1.
        # Running tally of absolute error. Can be positive or negative. Want to be 0.
        self.error_sum += self.target_accept - p_accept
        # This is the next proposed (log) step size. Note it is biased towards mu.
        log_step = self.mu - self.error_sum / (np.sqrt(self.t) * self.gamma)
        # Forgetting rate. As `t` gets bigger, `eta` gets smaller.
        eta = self.t ** -self.kappa
        # Smoothed average step size
        self.log_averaged_step = eta * log_step + (1 - eta) * self.log_averaged_step
        # This is a stateful update, so t keeps updating
        self.t += 1

        # Return both the noisy step size, and the smoothed step size
        return np.exp(log_step), np.exp(self.log_averaged_step)

    
    def __call__(self, i, p_accept):
        if i == 0:
            return self.initial_step_size 
        elif i < self.nadapt:
            step_size, avgstepsize = self.update(p_accept)
        elif i == self.nadapt:
            _, step_size = self.update(p_accept)
            print("\nStep size fixed to : %0.3e\n"%step_size)
        else:
            step_size = np.exp(self.log_averaged_step)
        return step_size


    
class HMC():

    def __init__(self, log_prob, grad_log_prob=None, log_prob_and_grad=None, invmetric_diag=None):

        self.log_prob, self.grad_log_prob = log_prob, grad_log_prob
        self.log_prob_and_grad = log_prob_and_grad
        if invmetric_diag is None: self.invmetric_diag = 1.
        else: self.invmetric_diag = invmetric_diag
        self.metricstd = self.invmetric_diag**-0.5

        assert not((self.grad_log_prob == None) & (self.log_prob_and_grad == None))
        #
        self.V = lambda x : self.log_prob(x)*-1.
        self.KE = lambda p: 0.5*(p**2 * self.invmetric_diag).sum()
        self.KE_g = lambda p: p * self.invmetric_diag
        #
        self.leapcount = 0 
        self.Vgcount = 0 
        self.Hcount = 0
 
   
    def V_g(self, x):
        self.Vgcount += 1
        if self.grad_log_prob is not None:
            v_g = self.grad_log_prob(x)
        elif self.log_prob_and_grad is not None:
            v, v_g = self.log_prob_and_grad(x)
        return v_g *-1.

        
    def V_vandg(self, x):
        if self.log_prob_and_grad is not None:
            self.Vgcount += 1
            v, v_g = self.log_prob_and_grad(x)
            return v*-1., v_g*-1
        else:
            raise NotImplementedError
        

    def unit_norm_KE(self, p):
        return 0.5 * (p**2).sum()


    def unit_norm_KE_g(self, p):
        return p


    def H(self, q, p, Vq=None):
        if Vq is None: 
            self.Hcount += 1
            Vq = self.V(q)
        return Vq + self.KE(p)


    def leapfrog(self, q, p, N, step_size):
        self.leapcount += 1 
        q0, p0 = q, p
        try:
            p = p - 0.5*step_size * self.V_g(q) 
            for i in range(N-1):
                q = q + step_size * self.KE_g(p)
                p = p - step_size * self.V_g(q) 
            q = q + step_size * self.KE_g(p)
            p = p - 0.5*step_size * self.V_g(q) 
            
            print(step_size, N)
            print("Cur Pos:\t", q0)
            print("Cur Mom:\t", p0)
            print("Prop Pos:\t", q) 
            print("Prop Mom:\t", p)
            return q, p

        except Exception as e:
            print("exception : ", e)
            return q0, p0


    def leapfrog_Vgq(self, q, p, N, step_size, V_q=None, V_gq=None):
        self.leapcount += 1 
        q0, p0, V_q0, V_gq0 = q, p, V_q, V_gq
        try:
            if V_gq is None:
                p = p - 0.5*step_size * self.V_g(q) 
            else:
                p = p - 0.5*step_size * V_gq
            for i in range(N-1):
                q = q + step_size * self.KE_g(p)
                p = p - step_size * self.V_g(q) 

            q = q + step_size * self.KE_g(p)
            if self.log_prob_and_grad is not None:
                V_q1, V_gq1 = self.V_vandg(q) 
            else:
                V_q1, V_gq1 = None, self.V_g(q) 
            p = p - 0.5*step_size * V_gq1
            return q, p, V_q1, V_gq1

        except Exception as e:
            print("exception : ", e)
            return q0, p0, V_q0, V_gq0


    def metropolis(self, qp0, qp1, V_q0=None, V_q1=None, u=None):
        
        q0, p0 = qp0
        q1, p1 = qp1
        H0 = self.H(q0, p0, V_q0)
        H1 = self.H(q1, p1, V_q1)
        prob = np.exp(H0 - H1)
        #prob = min(1., np.exp(H0 - H1))
        
        if u is None: u =  np.random.uniform(0., 1., size=1)
        if np.isnan(prob) or np.isinf(prob) or (q0-q1).sum()==0: 
            return q0, p0, -1, [H0, H1]
        elif  u > min(1., prob):
            return q0, p0, 0., [H0, H1]
        else: return q1, p1, 1., [H0, H1]


    def step(self, q, nleap, step_size, **kwargs):

        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        p = np.random.normal(size=q.size).reshape(q.shape) * self.metricstd
        q1, p1 = self.leapfrog(q, p, nleap, step_size)
        q, p, accepted, Hs = self.metropolis([q, p], [q1, p1])
        return q, p, accepted, Hs, [self.Hcount, self.Vgcount, self.leapcount]


    def _parse_kwargs_sample(self, **kwargs):

        self.nsamples = kwargs['nsamples']
        self.burnin = kwargs['burnin']
        self.step_size = kwargs['step_size']
        self.nleap = kwargs['nleap']


    def adapt_stepsize(self, q, epsadapt, **kwargs):
        print("Adapting step size for %d iterations"%epsadapt)
        step_size = self.step_size
        epsadapt_kernel = DualAveragingStepSize(step_size)
        self._parse_kwargs_sample(**kwargs)
        
        for i in range(epsadapt+1):
            qprev = q.copy()
            q, p, acc, Hs, count = self.step(q, self.nleap, step_size)
            if (qprev == q).all():
                prob = 0 
            else:
                prob = np.exp(Hs[0] - Hs[1])
            if i < epsadapt:
                if np.isnan(prob): prob = 0.
                if prob > 1: prob = 1.
                step_size, avgstepsize = epsadapt_kernel.update(prob)
            elif i == epsadapt:
                _, step_size = epsadapt_kernel.update(prob)
                print("Step size fixed to : ", step_size)
                self.step_size = step_size
        return q

    
    def sample(self, q, p=None, callback=None, skipburn=True, epsadapt=0, **kwargs):

        kw = kwargs
        self._parse_kwargs_sample(**kwargs)
        
        state = Sampler()

        if epsadapt:
            q = self.adapt_stepsize(q, epsadapt, **kwargs)
            
        for i in range(self.nsamples + self.burnin):
            q, p, acc, Hs, count = self.step(q, self.nleap, self.step_size)
            state.i += 1
            state.accepts.append(acc)
            if skipburn & (i > self.burnin):
                state.samples.append(q)
                state.Hs.append(Hs)
                state.counts.append(count)
                if callback is not None: callback(state)
            
        state.to_array()
        return state



#####
class GHMC(HMC):
    
    def __init__(self, log_prob, grad_log_prob=None, log_prob_and_grad=None, invmetric_diag=None):
        super(GHMC, self).__init__(log_prob, grad_log_prob, log_prob_and_grad, invmetric_diag=invmetric_diag)        


    def step(self, q, p, nleap, step_size, alpha, V_q=None, V_gq=None):

        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        pn = np.random.normal(size=q.size).reshape(q.shape) * self.metricstd
        p = alpha*p + np.sqrt(1 - alpha**2)*pn

        q1, p1, V_q1, V_gq1 = self.leapfrog_Vgq(q, p, nleap, step_size, V_q=V_q, V_gq=V_gq)
        q, p, accepted, Hs = self.metropolis([q, p], [q1, p1], V_q, V_q1)

        if accepted == 1: 
            V_q, V_gq = V_q1, V_gq1
        else :
            p = -p # Momentum is flipped upon rejection

        return q, p, V_q, V_gq, accepted, Hs, [self.Hcount, self.Vgcount, self.leapcount]


    def sample(self, q, p=None, callback=None, **kwargs):

        kw = kwargs
        self._parse_kwargs_sample(**kwargs)
        
        state = Sampler()
        if p is None: p = np.random.normal(0, 1, q.size).reshape(q.shape)
        V_q, V_gq = None, None
        for i in range(self.nsamples + self.burnin):
            q, p, V_q, V_gq, acc, Hs, count = self.step(q, p, V_q=V_q, V_gq=V_gq, 
                                                          alpha=kw['alpha'],
                                                          nleap=self.nleap,
                                                          step_size=self.step_size)
            state.i += 1
            state.samples.append(q)
            state.accepts.append(acc)
            state.Hs.append(Hs)
            state.counts.append(count)
            if callback is not None: callback(state)

        state.to_array()
        return state


#####
class Persistent_GHMC(HMC):
    
    def __init__(self, log_prob, grad_log_prob=None, log_prob_and_grad=None, invmetric_diag=None):
        super(Persistent_GHMC, self).__init__(log_prob, grad_log_prob, log_prob_and_grad, invmetric_diag=invmetric_diag)       


    def step(self, q, p, u, nleap, step_size, alpha, delta, V_q=None, V_gq=None):

        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        pn = np.random.normal(size=q.size).reshape(q.shape) * self.metricstd
        p = alpha*p + np.sqrt(1 - alpha**2)*pn
        u = (u + 1 + delta)%2 - 1 

        q1, p1, V_q1, V_gq1 = self.leapfrog_Vgq(q, p, nleap, step_size, V_q=V_q, V_gq=V_gq)
        q, p, accepted, Hs = self.metropolis([q, p], [q1, p1], V_q, V_q1, u)

        if accepted == 1: 
            V_q, V_gq = V_q1, V_gq1
            H0, H1 = Hs
            pfac = np.exp(H0 - H1)
            u = u/pfac
        else: 
            p = -p # Momentum is flipped upon rejection

        return q, p, u, V_q, V_gq, accepted, Hs, [self.Hcount, self.Vgcount, self.leapcount]


    def sample(self, q, p=None, callback=None, **kwargs):
        kw = kwargs
        self._parse_kwargs_sample(**kwargs)
        
        state = Sampler()
        state.u = []
        if p is None: p = np.random.normal(0, 1, q.size).reshape(q.shape)
        u = np.random.normal(0, 1)
        V_q, V_gq = None, None
        for i in range(self.nsamples + self.burnin):

            q, p, u, V_q, V_gq, acc, Hs, count = self.step(q, p, u, V_q=V_q, V_gq=V_gq, 
                                                    alpha=kw['alpha'], delta=kw['delta'],
                                                    nleap=self.nleap, step_size=self.step_size)
            state.i += 1
            state.samples.append(q)
            state.accepts.append(acc)
            state.Hs.append(Hs)
            state.counts.append(count)
            state.u.append(u)
            if callback is not None: callback(state)

        state.to_array()
        return state




#####
class DRHMC_legacy(HMC):
    '''vanilla DRHMC code
    '''
    def __init__(self, log_prob, grad_log_prob=None, log_prob_and_grad=None, invmetric_diag=None):
        super(DRHMC_legacy, self).__init__(log_prob, grad_log_prob, log_prob_and_grad, invmetric_diag=invmetric_diag)       


    def get_num(self, nprops, q, p, nleap, step_size, stepfactor): 

        avec = np.zeros(nprops)
        H0 = self.H(q, p)    
        for j in range(nprops):
            fac = stepfactor**(j)
            qj, pj = self.leapfrog(q, p, int(nleap*fac), step_size/fac)
            Hj = self.H(qj, pj)
            pfac = np.exp(H0 - Hj)
            if  (q - qj).sum()==0: 
                pfac = 0.
            if j:
                den = np.prod(1-avec[:j])
                num = self.get_num(j, qj, -pj, nleap, step_size, stepfactor)
                prob = pfac*num/den
            else: 
                prob = pfac
            if np.isnan(prob) or np.isinf(prob): 
                return 0. #np.nan
            else: avec[j] = min(1., prob)
            if np.prod(1-avec): pass
            else: 
                return np.prod(1-avec)
        return np.prod(1-avec)


    def step(self, q, nleap, step_size, nprops, stepfactor):
        
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        avec = np.zeros(nprops)

        q0 = q
        p0 = np.random.normal(size=q0.size).reshape(q0.shape) * self.metricstd
        H0 = self.H(q0, p0)

        #First step
        q1, p1 = self.leapfrog(q0, p0, nleap, step_size)
        H1 = self.H(q1, p1)
        pfac = np.exp(H0 - H1)
        prob = pfac
        if np.isnan(prob) or np.isinf(prob) or (q0 - q1).sum()==0: 
            prob = 0.
        prob = min(1., prob)
        avec[0] = prob

        acc = np.random.uniform()
        if  acc <= avec[0]: 
            return q1, p1, 1, avec, [self.Hcount, self.Vgcount, self.leapcount]

        else:
            #delayed proposals
            for j in range(1, nprops):
                fac = stepfactor**(j)
                nleapf, step_sizef = int(nleap*fac), step_size/fac
                qj, pj = self.leapfrog(q0, p0, nleapf, step_sizef)
                Hj = self.H(qj, pj)
                pfac = np.exp(H0 - Hj)
                if  (q0 - qj).sum()==0:
                    pfac = 0.
                den = np.prod(1-avec[:j])
                num = self.get_num(j, qj, -pj, nleap, step_size, stepfactor)
                prob = pfac*num/den
                if np.isnan(prob) or np.isinf(prob): prob = 0.
                avec[j] = min(1., prob)
                acc = np.random.uniform()
                if acc < avec[j]: 
                    return qj, pj, j+1, avec, [self.Hcount, self.Vgcount, self.leapcount]
            return q0, p0, 0, avec, [self.Hcount, self.Vgcount, self.leapcount]

        
    def sample(self, q, p=None, callback=None, **kwargs):
        kw = kwargs
        self._parse_kwargs_sample(**kwargs)
        
        state = Sampler()
        state.avecs = []
        if p is None: p = np.random.normal(0, 1, q.size).reshape(q.shape)

        for i in range(self.nsamples + self.burnin):

            q, p, acc, avecs, count = self.step(q, nleap=self.nleap, step_size=self.step_size, 
                                                nprops=kwargs['nprops'], stepfactor=kwargs['stepfactor'])
            state.i += 1
            state.samples.append(q)
            state.accepts.append(acc)
            state.avecs.append(avecs)
            state.counts.append(count)
            if callback is not None: callback(state)

        state.to_array()
        return state





#####
class DRHMC(HMC):
    '''vanilla DRHMC code
    '''
    def __init__(self, log_prob, grad_log_prob=None, log_prob_and_grad=None, invmetric_diag=None):
        super(DRHMC, self).__init__(log_prob, grad_log_prob, log_prob_and_grad, invmetric_diag=invmetric_diag)       


    def get_num(self, mj, q, p, nleap, step_size, stepfactor, H0): 

        avec = np.zeros(mj)
        #H0 = self.H(q, p)    
        for j in range(mj): 
            fac = stepfactor**(j)
            qj, pj = self.leapfrog(q, p, int(nleap*fac), step_size/fac)
            print("Ghost Pos:\t", qj)
            print("Ghost Mom:\t", pj)
            Hj = self.H(qj, pj)
            pfac = np.exp(H0 - Hj)
            if  (q - qj).sum()==0: 
                pfac = 0.
            if j: # if j > 0
                den = np.prod(1-avec[:j])
                num = self.get_num(j, qj, -pj, nleap, step_size, stepfactor, Hj)
                prob = pfac*num/den
            else: # if j = 0
                prob = pfac
            if np.isnan(prob) or np.isinf(prob): 
                return 0. #np.nan
            else:
                print("Prob:\t", prob)
                avec[j] = min(1., prob)
            if np.prod(1-avec): 
                pass
            else: 
                return np.prod(1-avec)
        return np.prod(1-avec)


    def step(self, q, p, nleap, step_size, nprops, stepfactor, rng):
        
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        avec = np.zeros(nprops)

        q0 = q
        p0 = np.random.normal(size=q0.size).reshape(q0.shape) * self.metricstd
        p0 = p
        H0 = self.H(q0, p0)

        for j in range(nprops):
            fac = stepfactor**(j)
            nleapf, step_sizef = int(nleap*fac), step_size/fac
            qj, pj = self.leapfrog(q0, p0, nleapf, step_sizef)
            Hj = self.H(qj, pj)
            pfac = np.exp(H0 - Hj)
            if  (q0 - qj).sum()==0:
                pfac = 0.
            den = np.prod(1-avec[:j])
            if q0[0] < -4:
                vb = True
                print(j, q0, qj)
            else: vb = False
            num = self.get_num(j, qj, -pj, nleap, step_size, stepfactor, Hj)
            prob = pfac*num/den
            print("Hamiltonian Error:\t", pfac)
            print("Num:\t", num)
            print("Den:\t", den)
            if np.isnan(prob) or np.isinf(prob): 
                prob = 0.
            avec[j] = min(1., prob)
            acc = np.random.uniform()
            print("Acceptance Prob:\t", avec[j], acc)
            if acc < avec[j]: 
                return qj, pj, j+1, avec, [self.Hcount, self.Vgcount, self.leapcount]
        return q0, p0, 0, avec, [self.Hcount, self.Vgcount, self.leapcount]


        
    def sample(self, q, p=None, callback=None, **kwargs):
        kw = kwargs
        self._parse_kwargs_sample(**kwargs)
        
        state = Sampler()
        state.avecs = []
        if p is None: p = np.random.normal(0, 1, q.size).reshape(q.shape)

        for i in range(self.nsamples + self.burnin):

            q, p, acc, avecs, count = self.step(q, p, nleap=self.nleap, step_size=self.step_size, 
                                                nprops=kwargs['nprops'], stepfactor=kwargs['stepfactor'], rng=kwargs['rng'])
            state.i += 1
            if i > 570: print(i, q, p, acc, avecs, count)
            state.samples.append(q)
            state.accepts.append(acc)
            state.avecs.append(avecs)
            state.counts.append(count)
            if callback is not None: callback(state)

        state.to_array()
        return state




#####
class DRGHMC(HMC):
    '''DRHMC + GHMC
    '''
    def __init__(self, log_prob, grad_log_prob=None, log_prob_and_grad=None, invmetric_diag=None):
        super(DRGHMC, self).__init__(log_prob, grad_log_prob, log_prob_and_grad, invmetric_diag=invmetric_diag)       


    def get_num(self, mj, q, p, nleap, step_size, stepfactor, H0): 

        avec = np.zeros(mj)
        #H0 = self.H(q, p)    
        for j in range(mj):
            fac = stepfactor**(j)
            qj, pj = self.leapfrog(q, p, int(nleap*fac), step_size/fac)
            Hj = self.H(qj, pj)
            pfac = np.exp(H0 - Hj)
            if  (q - qj).sum()==0: 
                pfac = 0.
            if j:
                den = np.prod(1-avec[:j])
                num = self.get_num(j, qj, -pj, nleap, step_size, stepfactor, Hj)
                prob = pfac*num/den
            else: 
                prob = pfac
            if np.isnan(prob) or np.isinf(prob): 
                return 0. #np.nan
            else: avec[j] = min(1., prob)
            if np.prod(1-avec): pass
            else: 
                return np.prod(1-avec)
        return np.prod(1-avec)


    def step(self, q, p, nleap, step_size, alpha, nprops, stepfactor):
        
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        avec = np.zeros(nprops)

        q0 = q
        pn = np.random.normal(size=q.size).reshape(q.shape) * self.metricstd
        p0 = alpha*p + np.sqrt(1 - alpha**2)*pn

        H0 = self.H(q0, p0)

        for j in range(nprops):
            fac = stepfactor**(j)
            nleapf, step_sizef = int(nleap*fac), step_size/fac
            qj, pj = self.leapfrog(q0, p0, nleapf, step_sizef)
            Hj = self.H(qj, pj)
            pfac = np.exp(H0 - Hj)
            if  (q0 - qj).sum()==0:
                pfac = 0.
            den = np.prod(1-avec[:j])
            num = self.get_num(j, qj, -pj, nleap, step_size, stepfactor, Hj)
            prob = pfac*num/den
            if np.isnan(prob) or np.isinf(prob): 
                prob = 0.
            avec[j] = min(1., prob)
            acc = np.random.uniform()
            if acc < avec[j]: 
                return qj, pj, j+1, avec, [self.Hcount, self.Vgcount, self.leapcount]
        return q0, p0, 0, avec, [self.Hcount, self.Vgcount, self.leapcount]


        
    def sample(self, q, p=None, callback=None, **kwargs):
        kw = kwargs
        self._parse_kwargs_sample(**kwargs)
        
        state = Sampler()
        state.avecs = []
        if p is None: p = np.random.normal(0, 1, q.size).reshape(q.shape)

        for i in range(self.nsamples + self.burnin):

            q, p, acc, avecs, count = self.step(q, p, nleap=self.nleap, step_size=self.step_size, alpha=kwargs['alpha'], 
                                                nprops=kwargs['nprops'], stepfactor=kwargs['stepfactor'])
            state.i += 1
            state.samples.append(q)
            state.accepts.append(acc)
            state.avecs.append(avecs)
            state.counts.append(count)
            if callback is not None: callback(state)

        state.to_array()
        return state




#####
class Prob_DRHMC_Metrop(HMC):
    '''Probabilistic DRHMC with p_try = 1 - \pi(prop)/\pi(init)
    i.e. without the balancing Hastings factor of delayed attempts
    '''    
    def __init__(self, log_prob, grad_log_prob, log_prob_and_grad=None, invmetric_diag=None):
        super(Prob_DRHMC_Metrop, self).__init__(log_prob, grad_log_prob, log_prob_and_grad, invmetric_diag=invmetric_diag) 


    def get_num(self, m, q, p, N, ss, stepfactor): 

        avec = np.zeros(m)
        ptries = np.ones(m)
        H0 = self.H(q, p)    
        for j in range(m):
            fac = stepfactor**(j)
            qj, pj = self.leapfrog(q, p, int(N*fac), ss/fac)
            Hj = self.H(qj, pj)
            pfac = np.exp(H0 - Hj)
            if np.isnan(pfac) or np.isinf(pfac): ptries[j] = 1 
            else: ptries[j] = 1 - min(1. , pfac)
            if  (q - qj).sum()==0: 
                pfac = 0.
            if j:
                den = np.prod(1-avec[:j])  * np.prod(ptries[:j])
                num = self.get_num(j, qj, -pj, N, ss, stepfactor)
                prob = pfac*num/den
            else: 
                prob = pfac
            if np.isnan(prob) or np.isinf(prob): 
                return 0. #np.nan
            else: avec[j] = min(1., prob)
            if np.prod(1-avec): pass
            else: 
                return np.prod(1-avec)
        return np.prod(1-avec) * np.prod(ptries)


    def multi_step(self, m, q0, N, ss, stepfactor):
        
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        p0 = np.random.normal(size=q0.size).reshape(q0.shape) * self.metricstd
        avec = np.zeros(m)
        ptries = np.ones(m)
        H0 = self.H(q0, p0)
        q1, p1 = self.leapfrog(q0, p0, N, ss)
        H1 = self.H(q1, p1)
        pfac = np.exp(H0 - H1)
        prob = pfac
        if np.isnan(prob) or np.isinf(prob) or (q0 - q1).sum()==0: 
            prob = 0.
        prob = min(1., prob)
        avec[0] = prob
        if np.isnan(pfac) or np.isinf(pfac): ptries[0] = 1 
        else: ptries[0] = 1 - min(1. , pfac)

        acc = np.random.uniform()
        if  acc <= avec[0]: 
            return q1, p1, 0, avec, [self.Hcount, self.Vgcount, self.leapcount], ptries
        else:
            for j in range(1, m):
                if np.random.uniform() > ptries[j-1]:
                    return q0, p0, -2, avec, [self.Hcount, self.Vgcount, self.leapcount], ptries
                fac = stepfactor**(j)
                qj, pj = self.leapfrog(q0, p0, int(N*fac), ss/fac)
                Hj = self.H(qj, pj)
                pfac = np.exp(H0 - Hj)
                if  (q0 - qj).sum()==0:
                    pfac = 0.
                den = np.prod(1-avec[:j]) * np.prod(ptries[:j])
                num = self.get_num(j, qj, -pj, N, ss, stepfactor)
                if np.isnan(pfac) or np.isinf(pfac): ptries[j] = 1
                else: ptries[j] = 1 - min(1. , pfac)
                prob = pfac*num/den
                if np.isnan(prob) or np.isinf(prob): prob = 0.
                avec[j] = min(1., prob)
                acc = np.random.uniform()
                if acc < avec[j]: 
                    return qj, pj, j, avec, [self.Hcount, self.Vgcount, self.leapcount], ptries
            return q0, p0, -1, avec, [self.Hcount, self.Vgcount, self.leapcount], ptries





#####
class Prob_DRHMC_Hastings(HMC):
    '''Probabilistic DRHMC with p_try = 1 - \alpha
    where \alpha was the acceptance probility of the last stage
    '''
    def __init__(self, log_prob, grad_log_prob, invmetric_diag=None):
        super(Prob_DRHMC_Hastings, self).__init__(log_prob, grad_log_prob, invmetric_diag=invmetric_diag) 


    def get_num(self, m, q, p, N, ss, stepfactor): 

        avec = np.zeros(m)
        ptries = np.ones(m)
        H0 = self.H(q, p)    
        for j in range(m):
            fac = stepfactor**(j)
            qj, pj = self.leapfrog(q, p, int(N*fac), ss/fac)
            Hj = self.H(qj, pj)
            pfac = np.exp(H0 - Hj)
            if  (q - qj).sum()==0: 
                pfac = 0.
            if j:
                den = np.prod(1-avec[:j])  * np.prod(ptries[:j])
                num = self.get_num(j, qj, -pj, N, ss, stepfactor)
                prob = pfac*num/den
            else: 
                prob = pfac
            if np.isnan(prob) or np.isinf(prob): 
                return 0. #np.nan
            else: avec[j] = min(1., prob)

            if np.isnan(pfac) or np.isinf(pfac): ptries[j] = 1 
            else: ptries[j] = 1 - avec[j]#min(1. , pfac) 

            if np.prod(1-avec): pass
            else: 
                return np.prod(1-avec)
        return np.prod(1-avec) * np.prod(ptries)


    def multi_step(self, m, q0, N, ss, stepfactor):

        
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        p0 = np.random.normal(size=q0.size).reshape(q0.shape) * self.metricstd
        avec = np.zeros(m)
        ptries = np.ones(m)
        H0 = self.H(q0, p0)
        q1, p1 = self.leapfrog(q0, p0, N, ss)
        H1 = self.H(q1, p1)
        pfac = np.exp(H0 - H1)
        prob = pfac
        if np.isnan(prob) or np.isinf(prob) or (q0 - q1).sum()==0: 
            prob = 0.
        prob = min(1., prob)
        avec[0] = prob
        if np.isnan(pfac) or np.isinf(pfac): ptries[0] = 1 
        else: ptries[0] = 1 - avec[0] #min(1. , pfac)

        acc = np.random.uniform()
        if  acc <= avec[0]: 
            return q1, p1, 0, avec, [self.Hcount, self.Vgcount, self.leapcount], ptries
        else:
            for j in range(1, m):
                if np.random.uniform() > ptries[j-1]:
                    return q0, p0, -2, avec, [self.Hcount, self.Vgcount, self.leapcount], ptries
                fac = stepfactor**(j)
                qj, pj = self.leapfrog(q0, p0, int(N*fac), ss/fac)
                Hj = self.H(qj, pj)
                pfac = np.exp(H0 - Hj)
                if  (q0 - qj).sum()==0:
                    pfac = 0.
                den = np.prod(1-avec[:j]) * np.prod(ptries[:j])
                num = self.get_num(j, qj, -pj, N, ss, stepfactor)
                prob = pfac*num/den
                if np.isnan(prob) or np.isinf(prob): prob = 0.
                avec[j] = min(1., prob)
                if np.isnan(pfac) or np.isinf(pfac): ptries[j] = 1
                else: ptries[j] = 1 - avec[j]
                acc = np.random.uniform()
                if acc < avec[j]: 
                    return qj, pj, j, avec, [self.Hcount, self.Vgcount, self.leapcount], ptries
            return q0, p0, -1, avec, [self.Hcount, self.Vgcount, self.leapcount], ptries




alg_dict = {
    'hmc' : HMC,
    'ghmc' : GHMC, 
    'pghmc' : Persistent_GHMC, 
    'drhmc' : DRHMC, 
    #'drhmc' : DRHMC_legacy, 
    'drghmc' : DRGHMC, 
    'pdrhmc_metrop' : Prob_DRHMC_Metrop, 
    'pdrhmc_hasting' : Prob_DRHMC_Hastings,
}