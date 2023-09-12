import torch 
class test(torch.nn.Module):
    def __init__(self,dreamtime_m12s12,min_step=1,max_step=1000):
        super().__init__()
        self.min_step=min_step
        self.max_step=max_step
        self.dreamtime_m12s12=dreamtime_m12s12
        self.device=torch.device("cpu")

        if self.dreamtime_m12s12 is not None:
            self.dreamtime_t2index=lambda t: (t-self.min_step).long() if torch.is_tensor(t) else int(t-self.min_step)
            self.dreamtime_index2t=lambda i: torch.tensor(i).to(self.device)+self.min_step if not torch.is_tensor(i) else i+self.min_step
            self.dreamtime_w_sum=sum([self.dreamtime_w(t) for t in range(self.min_step,self.max_step+1)])
            self.dreamtime_p=self.dreamtime_w_normalized=lambda t: self.dreamtime_w(t)/self.dreamtime_w_sum
            self.dreamtime_p_list=torch.tensor([self.dreamtime_p(t) for t in range(self.min_step,self.max_step)]).to(self.device)
            self.dreamtime_p_t2Tsum=lambda t: self.dreamtime_p_list[self.dreamtime_t2index(t):].sum()
            self.dreamtime_p_t2Tsum_lookup=torch.tensor([self.dreamtime_p_t2Tsum(t) for t in range(self.min_step,self.max_step+1)]).to(self.device)

            self.dreamtime_optimal_t=lambda train_ratio: self.dreamtime_index2t((self.dreamtime_p_t2Tsum_lookup-train_ratio).abs().argmin())

    def dreamtime_w(self,t):
        if not torch.is_tensor(t):
            t=torch.tensor(t)
        if self.dreamtime_m12s12 is not None:
            m1=self.dreamtime_m12s12[0]
            m2=self.dreamtime_m12s12[1]
            s1=self.dreamtime_m12s12[2]
            s2=self.dreamtime_m12s12[3]
        else:
            return 1.
        if t > m1:
            return torch.exp(-(t - m1)**2 / (2 * (s1**2))).to(self.device)
        elif m2 <= t <= m1:
            return torch.ones(t.shape)
        elif t < m2:
            return torch.exp(-(t - m2)**2 / (2 * (s2**2))).to(self.device)

tester=test([800,500,300,100])



t, optimal_t, dreamtime_w=[],[],[]

for i in range(1,1001):
    i=torch.tensor(i)
    optimal_t.append(tester.dreamtime_optimal_t(i/1000))
    dreamtime_w.append(tester.dreamtime_w(optimal_t[-1]))
    t.append(i)

import matplotlib.pyplot as plt

# draw t-optimal_t t-dreamtime_w in two subplots
fig,ax=plt.subplots(2,1)
ax[0].plot(t,optimal_t)
ax[1].plot(optimal_t,dreamtime_w)

# name the subplots
ax[0].set_title("iter-optimal_t")
ax[1].set_title("optimal_t-dreamtime_w")

# separate the subplots with more space
fig.tight_layout()


# save to dreamtime.jpg
plt.savefig("dreamtime.jpg")
    