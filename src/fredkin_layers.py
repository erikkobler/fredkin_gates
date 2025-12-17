from collections.abc import generator

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

#--------------------------------------------------------------
#--1.-- FredkinLayer with 3 inputs and 1 constant (=4!=24 permutations)
#--------------------------------------------------------------
class Fredkin24Layer(nn.Module):
    # Fredkin layer using one constant value z (usually z=1), thus using 4!=24 permutations
    # M: from differentiably defined Fredkin gate derived matrix for output compuation
    # order: u,x1,x2,z,ux1,ux2
    M = torch.tensor([
        [+1, 0, 0, 0, 0, 0],
        [ 0, 0 ,+1, 0 ,+1 ,-1],
        [ 0 ,+1, 0, 0 ,-1 ,+1],
        [ 0, 0 ,0, +1, 0, 0]
    ], dtype=torch.float32)

    def __init__(self, din :int, dout :int, device :str ='cpu' ,seed :int =None,random_connections:bool=True ,offset :int =1,wgts_initial='evenly', z=1.0, verb=False):
        super().__init__()
        assert dout %3 ==0 , "dout must be divisible by 3"
        self.din = din
        self.dout = dout
        self.num_gates = dout // 3
        self.device = device
        self.verb = verb
        self.z = torch.tensor(z, dtype=torch.float32)
        self.offset = offset
        self.wgts_initial = wgts_initial
        self.seed = seed
        #initialize connections from inputs to gates (exactly 3 inputs per fredkin gate)
        self.random_connections = random_connections
        if self.random_connections:
            din_ids = torch.arange(self.din,device=device)
            used_inputs = self.din
            if self.din > self.num_gates*3:
                din_ids = din_ids[torch.randperm(self.din,device=device)[:self.num_gates*3]]
                used_inputs = self.num_gates*3
            slot_ids = torch.randperm(self.num_gates*3,device=device)[:used_inputs]
            gate_ids = slot_ids//3
            pos_ids = slot_ids % 3
            self.connections = -torch.ones(self.num_gates,3,dtype=torch.long,device=device)
            self.connections[gate_ids,pos_ids] = din_ids
            for g in range(self.num_gates):
                existing = self.connections[g,self.connections[g]>=0].tolist()
                needed = 3-len(existing)
                if needed >0:
                    pool = list(set(range(used_inputs))-set(existing))
                    extra = torch.multinomial(torch.ones(len(pool)), needed,replacement=False)
                    self.connections[g,self.connections[g]<0] = torch.tensor([pool[i] for i in extra],device=device)

        else:
            starts = (torch.arange(self.num_gates, device=self.device)*3 + self.offset) % self.din
            self.register_buffer('connections',
                             ((starts.unsqueeze(1) + torch.arange(3, device=self.device)) % self.din).long())
        if self.verb: print("Initial connections from inputs to gates: ", self.connections)
        #initialize weights for the permutations (randomly)
        if self.wgts_initial == 'gaussian':
            if self.seed is not None:
                g = torch.Generator(device=device)
                g.manual_seed(self.seed)
                init_tensor = torch.randn(self.num_gates, 24, generator=g, device=device)
            else:
                init_tensor = torch.randn(self.num_gates, 24, device=device)
        elif self.wgts_initial == 'evenly':
            init_tensor = torch.ones((self.num_gates, 24), device=device) / 24
        else: raise NotImplementedError("Initial wgts_initial must be either 'gaussian' or 'evenly'")
        self.wgts = torch.nn.Parameter(init_tensor)

    def forward(self, x):
        # 1. Gather 3 inputs
        bs = x.shape[0]
        x_exp = x.unsqueeze(1).expand(-1, self.num_gates, -1)       # (bs, num_gates, din)
        idx = self.connections.unsqueeze(0).expand(bs, -1, -1) # (bs, num_gates, 3)
        inputs = x_exp.gather(2, idx)                          # (bs, num_gates, 3)
        if self.verb: print("#1. Gather 3 inputs -> inputs:\n", inputs)

        # 2. Add constant z such that network can use constant inputs
        inputs = torch.cat([inputs, self.z.expand(bs, self.num_gates, 1)], dim=2)
        if self.verb: print("#2. Add constant z -> inputs:\n", inputs)

        # 3. Unsqueeze
        inputs = inputs.unsqueeze(2)  # (bs, num_gates, 1, 4)
        if self.verb: print("#3. Unsqueeze -> inputs:\n", inputs)

        # 4. All permutations of 4 inputs (3+1 constant): 4!=24 permutations
        perm_indices = torch.tensor(list(itertools.permutations(range(4))) ,device=inputs.device)
        inputs = inputs.expand(-1, -1, 24, -1)
        inputs = inputs.gather(3,
                               perm_indices.unsqueeze(0).unsqueeze(0).expand(inputs.shape[0],
                                                                             inputs.shape[1], -1, -1)) # (bs, num_gates, 24, 4)
        if self.verb :print("#4. Permutations -> inputs:\n", inputs)
        # Input matrix so far for each sample, for each output neuron:
        #   u   x1 x2 r
        # 1: a   b  c  1
        # 2: a   b  1  c
        # 3:...
        ...
        # 24:1   c  b  a

        # 5. Interaction terms ux1 and ux2
        col5 = inputs[..., 0] * inputs[..., 1]
        col6 = inputs[..., 0] * inputs[..., 2]
        inputs = torch.cat([inputs, col5.unsqueeze(-1), col6.unsqueeze(-1)], dim=-1) # (bs,num_gates,24,6)
        if self.verb: print("#5. Add interactions -> inputs:\n", inputs)

        # 6. Transpose
        inputs = inputs.transpose(2, 3)             # (bs,num_gates,6,24)
        if self.verb :print("#6. Transpose -> inputs:\n", inputs)

        # 7. Matrix multiplication: Output that each permutation would yield on its own
        Signals = torch.matmul(self.M, inputs)      # (bs, num_gates, 4, 24)
        if self.verb: print("#7. Compute Signals -> Signals:\n", Signals)

        # 8. Apply learnable weights
        if self.training:
            wgts_prob = F.softmax(self.wgts, dim=1)
        else:
            wgts_prob = torch.zeros_like(self.wgts)
            wgts_prob.scatter_(1 ,self.wgts.argmax(-1 ,keepdim=True) ,1.0)

        wgts_exp = wgts_prob.unsqueeze(0).expand(Signals.shape[0], -1, -1)
        wgts_exp = wgts_exp.unsqueeze(-1)
        output = torch.matmul(Signals, wgts_exp)    # (bs,num_gates,4,1)
        if self.verb: print("#8. Weighted sum -> output:\n", output)

        # 9. Remove residual output
        output = output[:, :, :3, :]                # (bs,num_gates,3,1)
        if self.verb: print("#9. Drop residual s -> output:\n", output)

        # 10. Flatten
        output = output.reshape(output.shape[0], -1) # (bs,dout=num_gates*3)
        if self.verb: print("#10. Flatten -> output:\n", output)

        return output

#--------------------------------------------------------------
#--2.-- FredkinLayer with 3 inputs and 2 constants (=5!/2=60 permutations)
#--------------------------------------------------------------
class Fredkin60Layer(nn.Module):
    # Fredkin layer using one constant value z (usually z=1), thus using 4!=24 permutations
    # M: from differentiably defined Fredkin gate derived matrix for output compuation
    # order: u,x1,x2,r1,r2,ux1,ux2
    M = torch.tensor([
        [+1, 0,  0, 0, 0,  0,  0],
        [ 0, 0 ,+1, 0, 0 ,+1 ,-1],
        [ 0 ,+1, 0, 0, 0 ,-1 ,+1],
        [ 0,  0 ,0,+1, 0,  0,  0],
        [ 0,  0, 0, 0,+1,  0,  0]
    ], dtype=torch.float32)

    def __init__(self, din :int, dout :int, device :str ='cpu' ,seed :int =None,random_connections:bool=True ,offset :int =1,wgts_initial='evenly',z=1, verb=False):
        super().__init__()
        assert dout %3 ==0 , "dout must be divisible by 3"
        self.din = din
        self.dout = dout
        self.num_gates = dout // 3
        self.device = device
        self.verb = verb
        self.r1 = torch.tensor(1, dtype=torch.float32)
        self.r2 = torch.tensor(0, dtype=torch.float32)
        self.offset = offset
        self.wgts_initial = wgts_initial
        self.seed = seed
        #initialize connections from inputs to gates (exactly 3 inputs per fredkin gate)
        self.random_connections = random_connections
        if self.random_connections:
            din_ids = torch.arange(self.din,device=device)
            used_inputs = self.din
            if self.din > self.num_gates*3:
                din_ids = din_ids[torch.randperm(self.din,device=device)[:self.num_gates*3]]
                used_inputs = self.num_gates*3
            slot_ids = torch.randperm(self.num_gates*3,device=device)[:used_inputs]
            gate_ids = slot_ids//3
            pos_ids = slot_ids % 3
            self.connections = -torch.ones(self.num_gates,3,dtype=torch.long,device=device)
            self.connections[gate_ids,pos_ids] = din_ids
            for g in range(self.num_gates):
                existing = self.connections[g,self.connections[g]>=0].tolist()
                needed = 3-len(existing)
                if needed >0:
                    pool = list(set(range(used_inputs))-set(existing))
                    extra = torch.multinomial(torch.ones(len(pool)), needed,replacement=False)
                    self.connections[g,self.connections[g]<0] = torch.tensor([pool[i] for i in extra],device=device)

        else:
            starts = (torch.arange(self.num_gates, device=self.device)*3 + self.offset) % self.din
            self.register_buffer('connections',
                             ((starts.unsqueeze(1) + torch.arange(3, device=self.device)) % self.din).long())
        if self.verb: print("Initial connections from inputs to gates: ", self.connections)
        # initialize weights for the permutations (randomly)
        if self.wgts_initial == 'gaussian':
            if self.seed is not None:
                g = torch.Generator(device=device)
                g.manual_seed(self.seed)
                init_tensor = torch.randn(self.num_gates, 60, generator=g, device=device)
            else:
                init_tensor = torch.randn(self.num_gates, 60, device=device)
        elif self.wgts_initial == 'evenly':
            init_tensor = torch.ones((self.num_gates, 60), device=device) / 60
        else:
            raise NotImplementedError("Initial wgts_initial must be either 'gaussian' or 'evenly'")
        self.wgts = torch.nn.Parameter(init_tensor)

    def forward(self, x):
        # 1. Gather 3 inputs
        bs = x.shape[0]
        x_exp = x.unsqueeze(1).expand(-1, self.num_gates, -1)       # (bs, num_gates, din)
        idx = self.connections.unsqueeze(0).expand(bs, -1, -1) # (bs, num_gates, 3)
        inputs = x_exp.gather(2, idx)                          # (bs, num_gates, 3)
        if self.verb: print("#1. Gather 3 inputs -> inputs:\n", inputs)

        # 2. Add constant z such that network can use constant inputs
        inputs = torch.cat([inputs, self.r1.expand(bs, self.num_gates, 1),self.r2.expand(bs,self.num_gates,1)], dim=2)
        if self.verb: print("#2. Add constants r1=1 and r2=0 -> inputs:\n", inputs)

        # 3. Unsqueeze
        inputs = inputs.unsqueeze(2)  # (bs, num_gates, 1, 5)
        if self.verb: print("#3. Unsqueeze -> inputs:\n", inputs)

        # 4. All permutations of 4 inputs (3+2 constants): 5!/2=60 permutations
        perm_indices = torch.tensor(list(itertools.permutations(range(5))) ,device=inputs.device)
        perm_indices = perm_indices[::2]
        inputs = inputs.expand(-1, -1, 60, -1)
        inputs = inputs.gather(3,
                               perm_indices.unsqueeze(0).unsqueeze(0).expand(inputs.shape[0],
                                                                             inputs.shape[1], -1, -1)) # (bs, num_gates, 60, 5)
        inputs = inputs
        if self.verb :print("#4. Permutations -> inputs:\n", inputs)
        # Input matrix so far for each sample, for each output neuron:
        #   u   x1 x2 r1  r2
        # 1: a   b  c  1   0
        # 2: a   b  1  c   0
        # 3: a   b  0  c   1
        # 4: a   c  b  1   0
        ...
        # 60:0   c  b  a   1

        # 5. Interaction terms ux1 and ux2
        col6 = inputs[..., 0] * inputs[..., 1]
        col7 = inputs[..., 0] * inputs[..., 2]
        inputs = torch.cat([inputs, col6.unsqueeze(-1), col7.unsqueeze(-1)], dim=-1) # (bs,num_gates,60,7)
        if self.verb: print("#5. Add interactions -> inputs:\n", inputs)

        # 6. Transpose
        inputs = inputs.transpose(2, 3)             # (bs,num_gates,7,60)
        if self.verb :print("#6. Transpose -> inputs:\n", inputs)

        # 7. Matrix multiplication: Output that each permutation would yield on its own
        Signals = torch.matmul(self.M, inputs)      # (bs, num_gates, 5, 60)
        if self.verb: print("#7. Compute Signals -> Signals:\n", Signals)

        # 8. Apply learnable weights
        if self.training:
            wgts_prob = F.softmax(self.wgts, dim=1)
        else:
            wgts_prob = torch.zeros_like(self.wgts)
            wgts_prob.scatter_(1 ,self.wgts.argmax(-1 ,keepdim=True) ,1.0)

        wgts_exp = wgts_prob.unsqueeze(0).expand(Signals.shape[0], -1, -1)
        wgts_exp = wgts_exp.unsqueeze(-1)
        output = torch.matmul(Signals, wgts_exp)    # (bs,num_gates,5,1)
        if self.verb: print("#8. Weighted sum -> output:\n", output)

        # 9. Remove residual output
        output = output[:, :, :3, :]                # (bs,num_gates,3,1)
        if self.verb: print("#9. Drop residual s -> output:\n", output)

        # 10. Flatten
        output = output.reshape(output.shape[0], -1) # (bs,dout=num_gates*3)
        if self.verb: print("#10. Flatten -> output:\n", output)

        return output

#--------------------------------------------------------------
#--3.-- FredkinLayer with 3 inputs and learnable selector (stage1-weight) for each input (stage 2: 3!=6 permutations) =9 total params
#--------------------------------------------------------------
class Fredkin3plus6Layer(nn.Module):
    # Fredkin layer using 2 consecutive stages of weights: First weight adds learnable "selector" to input value
    # such that it gate may learn to ignore the input and replace it by 0 or 1.
    # Second, all 3!=6 permutations of the resulting intermediate inputs are used to train how gate is wired.
    # M: from differentiably defined Fredkin gate derived matrix for output computation
    # order: u,x1,x2,ux1,ux2
    M = torch.tensor([
        [+1, 0,  0,  0,  0],
        [ 0, 0 ,+1 ,+1 ,-1],
        [ 0 ,+1, 0,-1 ,+1]
    ], dtype=torch.float32)

    def __init__(self, din: int, dout: int, device: str = 'cpu', seed: int = None, random_connections: bool = True,
                 offset: int = 1,wgts_initial='evenly', z=1,crisp_cutoff=0.5, verb=False):
        super().__init__()
        assert dout % 3 == 0, "dout must be divisible by 3"
        self.din = din
        self.dout = dout
        self.num_gates = dout // 3
        self.device = device
        self.verb = verb
        self.offset = offset
        self.wgts_initial = wgts_initial
        self.crisp_cutoff = crisp_cutoff
        self.seed = seed
        self.eps = 1e-9  # small constant to prevent division by 0 in stage 1 of computation (#2.)
        # initialize connections from inputs to gates (exactly 3 inputs per fredkin gate)
        self.random_connections = random_connections
        if self.random_connections:
            g = None
            if self.seed is not None:
                g = torch.Generator(device=device)
                g.manual_seed(self.seed)
            din_ids = torch.arange(self.din, device=device)
            used_inputs = self.din
            if self.din > self.num_gates * 3:
                din_ids = din_ids[torch.randperm(self.din,generator=g, device=device)[:self.num_gates * 3]]
                used_inputs = self.num_gates * 3
            slot_ids = torch.randperm(self.num_gates * 3,generator=g, device=device)[:used_inputs]
            gate_ids = slot_ids // 3
            pos_ids = slot_ids % 3
            self.connections = -torch.ones(self.num_gates, 3, dtype=torch.long, device=device)
            self.connections[gate_ids, pos_ids] = din_ids
            for gate in range(self.num_gates):
                existing = self.connections[gate, self.connections[gate] >= 0].tolist()
                needed = 3 - len(existing)
                if needed > 0:
                    pool = list(set(range(used_inputs)) - set(existing))
                    extra = torch.multinomial(torch.ones(len(pool)), needed, replacement=False,generator=g)
                    self.connections[gate, self.connections[gate] < 0] = torch.tensor([pool[i] for i in extra], device=device)

        else:
            starts = (torch.arange(self.num_gates, device=self.device) * 3 + self.offset) % self.din
            self.register_buffer('connections',
                                 ((starts.unsqueeze(1) + torch.arange(3, device=self.device)) % self.din).long())
        if self.verb: print("Initial connections from inputs to gates: ", self.connections)

        #initialize "selector" for each input
        if self.seed is not None:
            g = torch.Generator(device=self.device)
            g.manual_seed(self.seed)
            init_tensor = torch.randn(self.num_gates, 3, generator=g, device=device)
        else:
            init_tensor = torch.randn(self.num_gates, 3, device=device)
            init_tensor = torch.zeros(self.num_gates, 3, device=device)+self.eps
        self.selector = torch.nn.Parameter(init_tensor, requires_grad=True)

        # initialize weights for the permutations (randomly)
        # initialize weights for the permutations (randomly)
        if self.wgts_initial == 'gaussian':
            if self.seed is not None:
                g = torch.Generator(device=device)
                g.manual_seed(self.seed)
                init_tensor = torch.randn(self.num_gates, 6, generator=g, device=device)
            else:
                init_tensor = torch.randn(self.num_gates, 6, device=device)
        elif self.wgts_initial == 'evenly':
            init_tensor = torch.ones((self.num_gates, 6), device=device) / 6
        else:
            raise NotImplementedError("Initial wgts_initial must be either 'gaussian' or 'evenly'")
        self.wgts = torch.nn.Parameter(init_tensor)

    def forward(self, x):
        # 1. Gather 3 inputs
        bs = x.shape[0]
        x_exp = x.unsqueeze(1).expand(-1, self.num_gates, -1)       # (bs, num_gates, din)
        idx = self.connections.unsqueeze(0).expand(bs, -1, -1) # (bs, num_gates, 3)
        inputs = x_exp.gather(2, idx)                          # (bs, num_gates, 3)
        if self.verb: print("#1. Gather 3 inputs -> inputs:\n", inputs)


        #2.(NEW) Compute intermediate inputs based on learnable parameters p1,p2,p3 for each gate (=stage 1)
        C = self.selector/(1+torch.abs(self.selector)) #softsign for smoother transitions
        #C = 2*torch.sigmoid(self.selector)-1.0 #(num_gates, 3) -> C is the actual selector ranging from [-1,1] based on learnable params
        if not self.training: #crisp choices for evaluation mode
            C = torch.where(C < -self.crisp_cutoff, -1.0,torch.where(C>self.crisp_cutoff,1.0, 0.0))
        C = C.unsqueeze(0).expand(bs,-1,-1) #(bs,num_gates,3)
        C_term1 = 1-torch.abs(C)
        C_term2 = torch.abs(C)
        inputs = C_term1 *inputs + (C+C_term2)/2 # every input now in range [0,1]
        if self.verb: print("#2. Compute intermediate input based on learnable selector -> inputs:\n", inputs)

        # 3. Unsqueeze
        inputs = inputs.unsqueeze(2)  # (bs, num_gates, 1, 3)
        if self.verb: print("#3. Unsqueeze -> inputs:\n", inputs)

        # 4. All permutations of 3 inputs: 3!=6 permutations
        perm_indices = torch.tensor(list(itertools.permutations(range(3))) ,device=inputs.device)
        inputs = inputs.expand(-1, -1, 6, -1)
        inputs = inputs.gather(3,
                               perm_indices.unsqueeze(0).unsqueeze(0).expand(inputs.shape[0],
                                                                             inputs.shape[1], -1, -1)) # (bs, num_gates, 6, 3)
        inputs = inputs
        if self.verb :print("#4. Permutations -> inputs:\n", inputs)
        # Input matrix so far for each sample, for each output neuron:
        #   u   x1 x2
        # 1: a   b  c
        # 2: a   c  b
        # 3: b   a  c
        # 4: b   c  a
        # 5: c   a  b
        # 6: c   b  a

        # 5. Interaction terms ux1 and ux2
        col4 = inputs[..., 0] * inputs[..., 1]
        col5 = inputs[..., 0] * inputs[..., 2]
        inputs = torch.cat([inputs, col4.unsqueeze(-1), col5.unsqueeze(-1)], dim=-1) # (bs,num_gates,6,5)
        if self.verb: print("#5. Add interactions -> inputs:\n", inputs)

        # 6. Transpose
        inputs = inputs.transpose(2, 3)             # (bs,num_gates,5,6)
        if self.verb :print("#6. Transpose -> inputs:\n", inputs)

        # 7. Matrix multiplication: Output that each permutation would yield on its own
        Signals = torch.matmul(self.M, inputs)      # (bs, num_gates, 3, 6)
        if self.verb: print("#7. Compute Signals -> Signals:\n", Signals)

        # 8. Apply learnable weights
        if self.training:
            wgts_prob = F.softmax(self.wgts, dim=1)
        else:
            wgts_prob = torch.zeros_like(self.wgts)
            wgts_prob.scatter_(1 ,self.wgts.argmax(-1 ,keepdim=True) ,1.0)

        wgts_exp = wgts_prob.unsqueeze(0).expand(Signals.shape[0], -1, -1)
        wgts_exp = wgts_exp.unsqueeze(-1)
        output = torch.matmul(Signals, wgts_exp)    # (bs,num_gates,3,1)
        if self.verb: print("#8. Weighted sum -> output:\n", output)

        # 9. Remove residual output (SKIPPED: In this FredkinLayer, there is no residual output to remove)
        #output = output[:, :, :3, :]                # (bs,num_gates,3,1)
        #if self.verb: print("#9. Drop residual s -> output:\n", output)

        # 10. Flatten
        output = output.reshape(output.shape[0], -1) # (bs,dout=num_gates*3)
        if self.verb: print("#10. Flatten -> output:\n", output)

        return output

#--------------------------------------------------------------
#--4.-- FredkinLayer with 3 inputs and no constants (=conservative), 3!=6 permutations/weights
#--------------------------------------------------------------
class Fredkin6Layer(nn.Module):
    # Fredkin layer using 3 inputs and 0 constants, thus 3!=6 permutations
    # order: u,x1,x2,ux1,ux2
    M = torch.tensor([
        [+1, 0,  0,  0,  0],
        [ 0, 0 ,+1 ,+1 ,-1],
        [ 0 ,+1, 0,-1 ,+1]
    ], dtype=torch.float32)

    def __init__(self, din: int, dout: int, device: str = 'cpu', seed: int = None, random_connections: bool = True,
                 offset: int = 1,wgts_initial='evenly', z=1,crisp_cutoff=0.5, verb=False):
        super().__init__()
        assert dout % 3 == 0, "dout must be divisible by 3"
        self.din = din
        self.dout = dout
        self.num_gates = dout // 3
        self.device = device
        self.verb = verb
        self.offset = offset
        self.wgts_initial = wgts_initial
        self.crisp_cutoff = crisp_cutoff
        self.seed = seed
        # initialize connections from inputs to gates (exactly 3 inputs per fredkin gate)
        self.random_connections = random_connections
        if self.random_connections:
            din_ids = torch.arange(self.din, device=device)
            used_inputs = self.din
            if self.din > self.num_gates * 3:
                din_ids = din_ids[torch.randperm(self.din, device=device)[:self.num_gates * 3]]
                used_inputs = self.num_gates * 3
            slot_ids = torch.randperm(self.num_gates * 3, device=device)[:used_inputs]
            gate_ids = slot_ids // 3
            pos_ids = slot_ids % 3
            self.connections = -torch.ones(self.num_gates, 3, dtype=torch.long, device=device)
            self.connections[gate_ids, pos_ids] = din_ids
            for g in range(self.num_gates):
                existing = self.connections[g, self.connections[g] >= 0].tolist()
                needed = 3 - len(existing)
                if needed > 0:
                    pool = list(set(range(used_inputs)) - set(existing))
                    extra = torch.multinomial(torch.ones(len(pool)), needed, replacement=False)
                    self.connections[g, self.connections[g] < 0] = torch.tensor([pool[i] for i in extra], device=device)

        else:
            starts = (torch.arange(self.num_gates, device=self.device) * 3 + self.offset) % self.din
            self.register_buffer('connections',
                                 ((starts.unsqueeze(1) + torch.arange(3, device=self.device)) % self.din).long())
        if self.verb: print("Initial connections from inputs to gates: ", self.connections)

        # initialize weights for the permutations (randomly)
        if self.wgts_initial == 'gaussian':
            if self.seed is not None:
                g = torch.Generator(device=device)
                g.manual_seed(self.seed)
                init_tensor = torch.randn(self.num_gates, 6, generator=g, device=device)
            else:
                init_tensor = torch.randn(self.num_gates, 6, device=device)
        elif self.wgts_initial == 'evenly':
            init_tensor = torch.ones((self.num_gates, 6), device=device) / 6
        else:
            raise NotImplementedError("Initial wgts_initial must be either 'gaussian' or 'evenly'")
        self.wgts = torch.nn.Parameter(init_tensor)

    def forward(self, x):
        # 1. Gather 3 inputs
        bs = x.shape[0]
        x_exp = x.unsqueeze(1).expand(-1, self.num_gates, -1)       # (bs, num_gates, din)
        idx = self.connections.unsqueeze(0).expand(bs, -1, -1) # (bs, num_gates, 3)
        inputs = x_exp.gather(2, idx)                          # (bs, num_gates, 3)
        if self.verb: print("#1. Gather 3 inputs -> inputs:\n", inputs)

        # 2. SKIPPED in this type of layer
        # 3. Unsqueeze
        inputs = inputs.unsqueeze(2)  # (bs, num_gates, 1, 3)
        if self.verb: print("#3. Unsqueeze -> inputs:\n", inputs)

        # 4. All permutations of 3 inputs: 3!=6 permutations
        perm_indices = torch.tensor(list(itertools.permutations(range(3))) ,device=inputs.device)
        inputs = inputs.expand(-1, -1, 6, -1)
        inputs = inputs.gather(3,
                               perm_indices.unsqueeze(0).unsqueeze(0).expand(inputs.shape[0],
                                                                             inputs.shape[1], -1, -1)) # (bs, num_gates, 6, 3)
        inputs = inputs
        if self.verb :print("#4. Permutations -> inputs:\n", inputs)
        # Input matrix so far for each sample, for each output neuron:
        #   u   x1 x2
        # 1: a   b  c
        # 2: a   c  b
        # 3: b   a  c
        # 4: b   c  a
        # 5: c   a  b
        # 6: c   b  a

        # 5. Interaction terms ux1 and ux2
        col4 = inputs[..., 0] * inputs[..., 1]
        col5 = inputs[..., 0] * inputs[..., 2]
        inputs = torch.cat([inputs, col4.unsqueeze(-1), col5.unsqueeze(-1)], dim=-1) # (bs,num_gates,6,5)
        if self.verb: print("#5. Add interactions -> inputs:\n", inputs)

        # 6. Transpose
        inputs = inputs.transpose(2, 3)             # (bs,num_gates,5,6)
        if self.verb :print("#6. Transpose -> inputs:\n", inputs)

        # 7. Matrix multiplication: Output that each permutation would yield on its own
        Signals = torch.matmul(self.M, inputs)      # (bs, num_gates, 3, 6)
        if self.verb: print("#7. Compute Signals -> Signals:\n", Signals)

        # 8. Apply learnable weights
        if self.training:
            wgts_prob = F.softmax(self.wgts, dim=1)
        else:
            wgts_prob = torch.zeros_like(self.wgts)
            wgts_prob.scatter_(1 ,self.wgts.argmax(-1 ,keepdim=True) ,1.0)

        wgts_exp = wgts_prob.unsqueeze(0).expand(Signals.shape[0], -1, -1)
        wgts_exp = wgts_exp.unsqueeze(-1)
        output = torch.matmul(Signals, wgts_exp)    # (bs,num_gates,3,1)
        if self.verb: print("#8. Weighted sum -> output:\n", output)

        # 9. Remove residual output (SKIPPED: In this FredkinLayer, there is no residual output to remove)
        #output = output[:, :, :3, :]                # (bs,num_gates,3,1)
        #if self.verb: print("#9. Drop residual s -> output:\n", output)

        # 10. Flatten
        output = output.reshape(output.shape[0], -1) # (bs,dout=num_gates*3)
        if self.verb: print("#10. Flatten -> output:\n", output)

        return output

#--------------------------------------------------------------
#--1.-- FredkinLayer with 2 inputs and 1 output
#--------------------------------------------------------------
class FredkinXLayer(nn.Module):
    # Fredkin layer not based on all permutations, but on specificially identified input configurations to compute logical functions
    # order: u,x1,x2,z,ux1,ux2
    M = torch.tensor([
        [+1, 0, 0, 0, 0, 0],
        [ 0, 0 ,+1, 0 ,+1 ,-1],
        [ 0 ,+1, 0, 0 ,-1 ,+1],
        [ 0, 0 ,0, +1, 0, 0]
    ], dtype=torch.float32)

    def __init__(self, din :int, dout :int, device :str ='cpu' ,seed :int =None,random_connections:bool=True ,offset :int =1,wgts_initial='evenly', z=1.0, verb=False):
        super().__init__()
        #assert dout %2 ==0 , "dout must be divisible by 2"
        self.din = din
        self.dout = dout
        self.num_gates = dout
        self.device = device
        self.verb = verb
        self.r1 = torch.tensor(1, dtype=torch.float32)
        self.r2 = torch.tensor(0, dtype=torch.float32)
        self.offset = offset
        self.wgts_initial = wgts_initial
        self.seed = seed
        self.num_permutations = 6
        #initialize connections from inputs to gates (exactly 3 inputs per fredkin gate)
        self.random_connections = random_connections
        if self.random_connections:
            din_ids = torch.arange(self.din,device=device)
            used_inputs = self.din
            if self.din > self.num_gates*2:
                din_ids = din_ids[torch.randperm(self.din,device=device)[:self.num_gates*2]]
                used_inputs = self.num_gates*2
            slot_ids = torch.randperm(self.num_gates*2,device=device)[:used_inputs]
            gate_ids = slot_ids//2
            pos_ids = slot_ids % 2
            self.connections = -torch.ones(self.num_gates,2,dtype=torch.long,device=device)
            self.connections[gate_ids,pos_ids] = din_ids
            for g in range(self.num_gates):
                existing = self.connections[g,self.connections[g]>=0].tolist()
                needed = 2-len(existing)
                if needed >0:
                    pool = list(set(range(used_inputs))-set(existing))
                    extra = torch.multinomial(torch.ones(len(pool)), needed,replacement=False)
                    self.connections[g,self.connections[g]<0] = torch.tensor([pool[i] for i in extra],device=device)

        else:
            starts = (torch.arange(self.num_gates, device=self.device)*2 + self.offset) % self.din
            self.register_buffer('connections',
                             ((starts.unsqueeze(1) + torch.arange(2, device=self.device)) % self.din).long())
        if self.verb: print("Initial connections from inputs to gates: ", self.connections)
        #initialize weights for the permutations (randomly)
        if self.wgts_initial == 'gaussian':
            if self.seed is not None:
                g = torch.Generator(device=device)
                g.manual_seed(self.seed)
                init_tensor = torch.randn(self.num_gates, self.num_permutations, generator=g, device=device)
            else:
                init_tensor = torch.randn(self.num_gates, self.num_permutations, device=device)
        elif self.wgts_initial == 'evenly':
            init_tensor = torch.ones((self.num_gates, self.num_permutations), device=device) / self.num_gates
        else: raise NotImplementedError("Initial wgts_initial must be either 'gaussian' or 'evenly'")
        self.wgts = torch.nn.Parameter(init_tensor)

    def forward(self, x):
        # 1. Gather 2 inputs
        bs = x.shape[0]
        x_exp = x.unsqueeze(1).expand(-1, self.num_gates, -1)       # (bs, num_gates, din)
        idx = self.connections.unsqueeze(0).expand(bs, -1, -1) # (bs, num_gates, 2)
        inputs = x_exp.gather(2, idx)                          # (bs, num_gates, 2)
        if self.verb: print("#1. Gather 2 inputs -> inputs:\n", inputs)

        # 2. Add constant z such that network can use constant inputs
        inputs = torch.cat([inputs, self.r1.expand(bs, self.num_gates, 1), self.r2.expand(bs, self.num_gates, 1)],dim=2)
        if self.verb: print("#2. Add constants r1,r2 -> inputs:\n", inputs)

        # 3. Unsqueeze
        inputs = inputs.unsqueeze(2)  # (bs, num_gates, 1, 4)
        if self.verb: print("#3. Unsqueeze -> inputs:\n", inputs)

        # 4. Permutations of interest
        perm_indices = torch.tensor([
            [0,2,3,3], #0 negate a, drop b
            [1,3,2,3], #1 negate b, drop a
            #[0,0,1,3], #2 bitsort a and b, with a as swap element (min)
            #[1,0,1,3], #3 bitsort b and a, with b as swap element (max)
            [0,3,1,3], #4 logical 'and' and -(b->a), with a as swap element #
            #[1,0,3,3], #5 logical 'and' and -(a->b), with b as swap element #
            [3,0,1,3], #6 swap a and b
            [2,0,1,3], #7 passthrough a and b
            [0,1,2,3], #8 logical 'or' and 'a->b'
            #[1,2,0,3], #9 logical 'or' and 'b->a'
            #[2,2,2,3], #10 pass 'true', drop a and b
            #[3,3,3,3], #11 pass 'false', drop a and b
            #[0,3,1,3],
            #[1,3,0,3]
        ])
        #perm_indices = torch.tensor(list(itertools.permutations(range(4))) ,device=inputs.device)
        inputs = inputs.expand(-1, -1, self.num_permutations, -1)
        inputs = inputs.gather(3,
                               perm_indices.unsqueeze(0).unsqueeze(0).expand(inputs.shape[0],
                                                                             inputs.shape[1], -1, -1)) # (bs, num_gates, 9, 4)
        if self.verb :print("#4. Permutations -> inputs:\n", inputs)
        # Input matrix so far for each sample, for each output neuron:
        #   u   x1 x2 r
        # 1: a   b  c  1
        # 2: a   b  1  c
        # 3:...
        ...
        # 24:1   c  b  a

        # 5. Interaction terms ux1 and ux2
        col5 = inputs[..., 0] * inputs[..., 1]
        col6 = inputs[..., 0] * inputs[..., 2]
        inputs = torch.cat([inputs, col5.unsqueeze(-1), col6.unsqueeze(-1)], dim=-1) # (bs,num_gates,9,6)
        if self.verb: print("#5. Add interactions -> inputs:\n", inputs)

        # 6. Transpose
        inputs = inputs.transpose(2, 3)             # (bs,num_gates,6,9)
        if self.verb :print("#6. Transpose -> inputs:\n", inputs)

        # 7. Matrix multiplication: Output that each permutation would yield on its own
        Signals = torch.matmul(self.M, inputs)      # (bs, num_gates, 4, 9)
        if self.verb: print("#7. Compute Signals -> Signals:\n", Signals)

        # 8. Apply learnable weights
        if self.training:
            wgts_prob = F.softmax(self.wgts, dim=1)
            #print("wgts", wgts_prob)
        else:
            wgts_prob = torch.zeros_like(self.wgts)
            wgts_prob.scatter_(1 ,self.wgts.argmax(-1 ,keepdim=True) ,1.0)

        wgts_exp = wgts_prob.unsqueeze(0).expand(Signals.shape[0], -1, -1)
        wgts_exp = wgts_exp.unsqueeze(-1)
        #print("exp:",wgts_exp)
        output = torch.matmul(Signals, wgts_exp)    # (bs,num_gates,4,1)
        if self.verb: print("#8. Weighted sum -> output:\n", output)

        # 9. Remove residual output and output (only pass 2 outputs on)
        #output = output[:, :, 1:3, :]                # (bs,num_gates,2,1)
        #if self.verb: print("#9. Drop residual s -> output:\n", output)
        #print(output)
        output = output[:,:,2,:]
        #print(output)

        # 10. Flatten
        output = output.reshape(output.shape[0], -1) # (bs,dout=num_gates*1)
        if self.verb: print("#10. Flatten -> output:\n", output)
        #print(output)

        #print("signals: ",Signals)
        #print("output: ",output)
        return output

    # --------------------------------------------------------------
    # --1.-- FredkinLayer with 2 inputs
    # --------------------------------------------------------------


class FredkinDepXLayer(nn.Module):
    # Fredkin layer not based on all permutations, but on specificially identified input configurations to compute logical functions
    # order: u,x1,x2,z,ux1,ux2
    M = torch.tensor([
        [+1, 0, 0, 0, 0, 0],
        [0, 0, +1, 0, +1, -1],
        [0, +1, 0, 0, -1, +1],
        [0, 0, 0, +1, 0, 0]
    ], dtype=torch.float32)

    def __init__(self, din: int, dout: int, device: str = 'cpu', seed: int = None, random_connections: bool = True,
                 offset: int = 1, wgts_initial='evenly', z=1.0, verb=False):
        super().__init__()
        # assert dout %2 ==0 , "dout must be divisible by 2"
        self.din = din
        self.dout = dout
        self.num_gates = dout
        self.device = device
        self.verb = verb
        self.r1 = torch.tensor(1, dtype=torch.float32)
        self.r2 = torch.tensor(0, dtype=torch.float32)
        self.offset = offset
        self.wgts_initial = wgts_initial
        self.seed = seed
        self.num_permutations = 6
        # initialize connections from inputs to gates (exactly 3 inputs per fredkin gate)
        self.random_connections = random_connections
        if self.random_connections:
            din_ids = torch.arange(self.din, device=device)
            used_inputs = self.din
            if self.din > self.num_gates * 2:
                din_ids = din_ids[torch.randperm(self.din, device=device)[:self.num_gates * 2]]
                used_inputs = self.num_gates * 2
            slot_ids = torch.randperm(self.num_gates * 2, device=device)[:used_inputs]
            gate_ids = slot_ids // 2
            pos_ids = slot_ids % 2
            self.connections = -torch.ones(self.num_gates, 2, dtype=torch.long, device=device)
            self.connections[gate_ids, pos_ids] = din_ids
            for g in range(self.num_gates):
                existing = self.connections[g, self.connections[g] >= 0].tolist()
                needed = 2 - len(existing)
                if needed > 0:
                    pool = list(set(range(used_inputs)) - set(existing))
                    extra = torch.multinomial(torch.ones(len(pool)), needed, replacement=False)
                    self.connections[g, self.connections[g] < 0] = torch.tensor([pool[i] for i in extra], device=device)

        else:
            starts = (torch.arange(self.num_gates, device=self.device) * 2 + self.offset) % self.din
            self.register_buffer('connections',
                                 ((starts.unsqueeze(1) + torch.arange(2, device=self.device)) % self.din).long())
        if self.verb: print("Initial connections from inputs to gates: ", self.connections)
        # initialize weights for the permutations (randomly)
        if self.wgts_initial == 'gaussian':
            if self.seed is not None:
                g = torch.Generator(device=device)
                g.manual_seed(self.seed)
                init_tensor = torch.randn(self.num_gates, self.num_permutations, generator=g, device=device)
            else:
                init_tensor = torch.randn(self.num_gates, self.num_permutations, device=device)
        elif self.wgts_initial == 'evenly':
            init_tensor = torch.ones((self.num_gates, self.num_permutations), device=device) / self.num_gates
        else:
            raise NotImplementedError("Initial wgts_initial must be either 'gaussian' or 'evenly'")
        self.wgts = torch.nn.Parameter(init_tensor)

    def forward(self, x):
        # 1. Gather 2 inputs
        bs = x.shape[0]
        x_exp = x.unsqueeze(1).expand(-1, self.num_gates, -1)  # (bs, num_gates, din)
        idx = self.connections.unsqueeze(0).expand(bs, -1, -1)  # (bs, num_gates, 2)
        inputs = x_exp.gather(2, idx)  # (bs, num_gates, 2)
        if self.verb: print("#1. Gather 2 inputs -> inputs:\n", inputs)

        # 2. Add constant z such that network can use constant inputs
        inputs = torch.cat([inputs, self.r1.expand(bs, self.num_gates, 1), self.r2.expand(bs, self.num_gates, 1)],
                           dim=2)
        if self.verb: print("#2. Add constants r1,r2 -> inputs:\n", inputs)

        # 3. Unsqueeze
        inputs = inputs.unsqueeze(2)  # (bs, num_gates, 1, 4)
        if self.verb: print("#3. Unsqueeze -> inputs:\n", inputs)

        # 4. Permutations of interest
        perm_indices = torch.tensor([
            [0, 2, 3, 3],  # 0 negate a, drop b
            [1, 3, 2, 3],  # 1 negate b, drop a
            # [0,0,1,3], #2 bitsort a and b, with a as swap element (min)
            # [1,0,1,3], #3 bitsort b and a, with b as swap element (max)
            [0, 3, 1, 3],  # 4 logical 'and' and -(b->a), with a as swap element #
            # [1,0,3,3], #5 logical 'and' and -(a->b), with b as swap element #
            [3, 0, 1, 3],  # 6 swap a and b
            [2, 0, 1, 3],  # 7 passthrough a and b
            [0, 1, 2, 3],  # 8 logical 'or' and 'a->b'
            # [1,2,0,3], #9 logical 'or' and 'b->a'
            # [2,2,2,3], #10 pass 'true', drop a and b
            # [3,3,3,3], #11 pass 'false', drop a and b
            # [0,3,1,3],
            # [1,3,0,3]
        ])
        # perm_indices = torch.tensor(list(itertools.permutations(range(4))) ,device=inputs.device)
        inputs = inputs.expand(-1, -1, self.num_permutations, -1)
        inputs = inputs.gather(3,
                               perm_indices.unsqueeze(0).unsqueeze(0).expand(inputs.shape[0],
                                                                             inputs.shape[1], -1,
                                                                             -1))  # (bs, num_gates, 9, 4)
        if self.verb: print("#4. Permutations -> inputs:\n", inputs)
        # Input matrix so far for each sample, for each output neuron:
        #   u   x1 x2 r
        # 1: a   b  c  1
        # 2: a   b  1  c
        # 3:...
        ...
        # 24:1   c  b  a

        # 5. Interaction terms ux1 and ux2
        col5 = inputs[..., 0] * inputs[..., 1]
        col6 = inputs[..., 0] * inputs[..., 2]
        inputs = torch.cat([inputs, col5.unsqueeze(-1), col6.unsqueeze(-1)], dim=-1)  # (bs,num_gates,9,6)
        if self.verb: print("#5. Add interactions -> inputs:\n", inputs)

        # 6. Transpose
        inputs = inputs.transpose(2, 3)  # (bs,num_gates,6,9)
        if self.verb: print("#6. Transpose -> inputs:\n", inputs)

        # 7. Matrix multiplication: Output that each permutation would yield on its own
        Signals = torch.matmul(self.M, inputs)  # (bs, num_gates, 4, 9)
        if self.verb: print("#7. Compute Signals -> Signals:\n", Signals)

        # 8. Apply learnable weights
        if self.training:
            wgts_prob = F.softmax(self.wgts, dim=1)
            # print("wgts", wgts_prob)
        else:
            wgts_prob = torch.zeros_like(self.wgts)
            wgts_prob.scatter_(1, self.wgts.argmax(-1, keepdim=True), 1.0)

        wgts_exp = wgts_prob.unsqueeze(0).expand(Signals.shape[0], -1, -1)
        wgts_exp = wgts_exp.unsqueeze(-1)
        # print("exp:",wgts_exp)
        output = torch.matmul(Signals, wgts_exp)  # (bs,num_gates,4,1)
        if self.verb: print("#8. Weighted sum -> output:\n", output)

        # 9. Remove residual output and output (only pass 2 outputs on)
        output = output[:, :, 1:3, :]                # (bs,num_gates,2,1)
        if self.verb: print("#9. Drop residual s -> output:\n", output)
        # print(output)
        #output = output[:, :, 2, :]
        # print(output)

        # 10. Flatten
        output = output.reshape(output.shape[0], -1)  # (bs,dout=num_gates*2)
        if self.verb: print("#10. Flatten -> output:\n", output)
        # print(output)

        # print("signals: ",Signals)
        # print("output: ",output)
        return output


#--------------------------------------------------------------
#--1.-- FredkinLayer with 3 inputs and 1 constant (=4!=24 permutations)
#--------------------------------------------------------------
class FredkinI12Layer(nn.Module):
    # Fredkin layer using one constant value z (usually z=1), thus using 4!=24 permutations
    # M: from differentiably defined Fredkin gate derived matrix for output compuation
    # order: u,x1,x2,ux1,ux2
    M = torch.tensor([
        [+1, 0, 0, 0, 0],
        [ 0, 0 ,+1 ,+1 ,-1],
        [ 0 ,+1, 0 ,-1 ,+1]
    ], dtype=torch.float32)

    def __init__(self, din :int, dout :int, device :str ='cpu' ,seed :int =None,random_connections:bool=True ,offset :int =1,wgts_initial='evenly', z=1.0, verb=False):
        super().__init__()
        assert dout %3 ==0 , "dout must be divisible by 3"
        self.din = din
        self.dout = dout
        self.num_gates = dout // 3
        self.device = device
        self.verb = verb
        self.z = torch.tensor(z, dtype=torch.float32)
        self.offset = offset
        self.wgts_initial = wgts_initial
        self.seed = seed
        #initialize connections from inputs to gates (exactly 3 inputs per fredkin gate)
        self.random_connections = random_connections
        if self.random_connections:
            din_ids = torch.arange(self.din,device=device)
            used_inputs = self.din
            if self.din > self.num_gates*3:
                din_ids = din_ids[torch.randperm(self.din,device=device)[:self.num_gates*3]]
                used_inputs = self.num_gates*3
            slot_ids = torch.randperm(self.num_gates*3,device=device)[:used_inputs]
            gate_ids = slot_ids//3
            pos_ids = slot_ids % 3
            self.connections = -torch.ones(self.num_gates,3,dtype=torch.long,device=device)
            self.connections[gate_ids,pos_ids] = din_ids
            for g in range(self.num_gates):
                existing = self.connections[g,self.connections[g]>=0].tolist()
                needed = 3-len(existing)
                if needed >0:
                    pool = list(set(range(used_inputs))-set(existing))
                    extra = torch.multinomial(torch.ones(len(pool)), needed,replacement=False)
                    self.connections[g,self.connections[g]<0] = torch.tensor([pool[i] for i in extra],device=device)

        else:
            starts = (torch.arange(self.num_gates, device=self.device)*3 + self.offset) % self.din
            self.register_buffer('connections',
                             ((starts.unsqueeze(1) + torch.arange(3, device=self.device)) % self.din).long())
        if self.verb: print("Initial connections from inputs to gates: ", self.connections)
        #initialize independent weights for all inputs
        if self.wgts_initial == 'gaussian':
            if self.seed is not None:
                g = torch.Generator(device=device)
                g.manual_seed(self.seed)
                init_tensor = torch.randn(self.num_gates, 3,4, generator=g, device=device)
            else:
                init_tensor = torch.randn(self.num_gates, 3,4, device=device)
        elif self.wgts_initial == 'evenly':
            init_tensor = torch.ones((self.num_gates, 3,4), device=device) / 4
        else: raise NotImplementedError("Initial wgts_initial must be either 'gaussian' or 'evenly'")
        self.wgts = torch.nn.Parameter(init_tensor)

    def forward(self, x):
        # 1. Gather 3 inputs
        bs = x.shape[0]
        x_exp = x.unsqueeze(1).expand(-1, self.num_gates, -1)       # (bs, num_gates, din)
        idx = self.connections.unsqueeze(0).expand(bs, -1, -1) # (bs, num_gates, 3)
        inputs = x_exp.gather(2, idx)                          # (bs, num_gates, 3)
        if self.verb: print("#1. Gather 3 inputs -> inputs:\n", inputs)

        # 2. Add constant z such that network can use constant inputs
        inputs = torch.cat([inputs, self.z.expand(bs, self.num_gates, 1)], dim=2) #(bs,num_gates,4)
        if self.verb: print("#2. Add constant z -> inputs:\n", inputs)

        # 3. Multiply weight matrix with inputs after passing weights through softmax
        if self.training:
            wgts_prob = F.softmax(self.wgts, dim=-1)                  #(num_gates,3,4)
        else:
            max_idx = self.wgts.argmax(dim=-1, keepdim=True)
            wgts_prob = torch.zeros_like(self.wgts)
            wgts_prob.scatter_(-1 ,max_idx ,1.0)
        inputs = torch.matmul(wgts_prob, inputs.unsqueeze(-1)).squeeze(-1)    #(
        #inputs = torch.einsum('gok,bgi->bgo', wgts_prob, inputs) #(bs,num_gates,3)
        #print("wgts_prob: ", wgts_prob)
        #print(inputs)

        # 4. Add interaction terms:
        v4 = inputs[..., 0] * inputs[..., 1]
        v5 = inputs[..., 0] * inputs[..., 2]
        inputs = torch.cat([inputs, v4.unsqueeze(-1), v5.unsqueeze(-1)], dim=-1) #(bs,num_gates,5)

        #5. Unsqueeze
        inputs = inputs.unsqueeze(-1) #(bs,num_gates,5,1)

        #6.Compute Fredkin output
        Signals = torch.matmul(self.M, inputs).squeeze(-1) #(bs,num_gates,3)

        #7. only keep third output of each gate
        Signals = Signals[:, :, ::3] #(bs,num_gates,1)

        # 10. Flatten
        output = Signals.reshape(Signals.shape[0], -1) # (bs,dout=num_gates*3)
        if self.verb: print("#10. Flatten -> output:\n", output)

        return output

#--------------------------------------------------------------
#--3.-- FredkinLayer with 3 inputs and learnable selector (stage1-weight) for each input (stage 2: 3!=6 permutations) =9 total params
#--------------------------------------------------------------
class FredkinInd3plus6Layer(nn.Module):
    # Fredkin layer using 2 consecutive stages of weights: First weight adds learnable "selector" to input value
    # such that it gate may learn to ignore the input and replace it by 0 or 1.
    # Second, all 3!=6 permutations of the resulting intermediate inputs are used to train how gate is wired.
    # M: from differentiably defined Fredkin gate derived matrix for output computation
    # order: u,x1,x2,ux1,ux2
    M = torch.tensor([
        [+1, 0,  0,  0,  0],
        [ 0, 0 ,+1 ,+1 ,-1],
        [ 0 ,+1, 0,-1 ,+1]
    ], dtype=torch.float32)

    def __init__(self, din: int, dout: int, device: str = 'cpu', seed: int = None, random_connections: bool = True,
                 offset: int = 1,wgts_initial='evenly', z=1,crisp_cutoff=0.5, verb=False):
        super().__init__()
        assert dout % 1 == 0, "dout must be divisible by 1"
        self.din = din
        self.dout = dout
        self.num_gates = dout
        self.device = device
        self.verb = verb
        self.offset = offset
        self.wgts_initial = wgts_initial
        self.crisp_cutoff = crisp_cutoff
        self.seed = seed
        self.eps = 1e-9  # small constant to prevent division by 0 in stage 1 of computation (#2.)
        # initialize connections from inputs to gates (exactly 3 inputs per fredkin gate)
        self.random_connections = random_connections
        if self.random_connections:
            g = None
            if self.seed is not None:
                g = torch.Generator(device=device)
                g.manual_seed(self.seed)
            din_ids = torch.arange(self.din, device=device)
            used_inputs = self.din
            if self.din > self.num_gates * 3:
                din_ids = din_ids[torch.randperm(self.din,generator=g, device=device)[:self.num_gates * 3]]
                used_inputs = self.num_gates * 3
            slot_ids = torch.randperm(self.num_gates * 3,generator=g, device=device)[:used_inputs]
            gate_ids = slot_ids // 3
            pos_ids = slot_ids % 3
            self.connections = -torch.ones(self.num_gates, 3, dtype=torch.long, device=device)
            self.connections[gate_ids, pos_ids] = din_ids
            for gate in range(self.num_gates):
                existing = self.connections[gate, self.connections[gate] >= 0].tolist()
                needed = 3 - len(existing)
                if needed > 0:
                    pool = list(set(range(used_inputs)) - set(existing))
                    extra = torch.multinomial(torch.ones(len(pool)), needed, replacement=False,generator=g)
                    self.connections[gate, self.connections[gate] < 0] = torch.tensor([pool[i] for i in extra], device=device)

        else:
            starts = (torch.arange(self.num_gates, device=self.device) * 3 + self.offset) % self.din
            self.register_buffer('connections',
                                 ((starts.unsqueeze(1) + torch.arange(3, device=self.device)) % self.din).long())
        if self.verb: print("Initial connections from inputs to gates: ", self.connections)

        #initialize "selector" for each input
        if self.seed is not None:
            g = torch.Generator(device=self.device)
            g.manual_seed(self.seed)
            init_tensor = torch.randn(self.num_gates, 3, generator=g, device=device)
        else:
            init_tensor = torch.randn(self.num_gates, 3, device=device)
            init_tensor = torch.zeros(self.num_gates, 3, device=device)+self.eps
        self.selector = torch.nn.Parameter(init_tensor, requires_grad=True)

        # initialize weights for the permutations (randomly)
        # initialize weights for the permutations (randomly)
        if self.wgts_initial == 'gaussian':
            if self.seed is not None:
                g = torch.Generator(device=device)
                g.manual_seed(self.seed)
                init_tensor = torch.randn(self.num_gates, 6, generator=g, device=device)
            else:
                init_tensor = torch.randn(self.num_gates, 6, device=device)
        elif self.wgts_initial == 'evenly':
            init_tensor = torch.ones((self.num_gates, 6), device=device) / 6
        else:
            raise NotImplementedError("Initial wgts_initial must be either 'gaussian' or 'evenly'")
        self.wgts = torch.nn.Parameter(init_tensor)

    def forward(self, x):
        # 1. Gather 3 inputs
        bs = x.shape[0]
        x_exp = x.unsqueeze(1).expand(-1, self.num_gates, -1)       # (bs, num_gates, din)
        idx = self.connections.unsqueeze(0).expand(bs, -1, -1) # (bs, num_gates, 3)
        inputs = x_exp.gather(2, idx)                          # (bs, num_gates, 3)
        if self.verb: print("#1. Gather 3 inputs -> inputs:\n", inputs)


        #2.(NEW) Compute intermediate inputs based on learnable parameters p1,p2,p3 for each gate (=stage 1)
        C = self.selector/(1+torch.abs(self.selector)) #softsign for smoother transitions
        #C = 2*torch.sigmoid(self.selector)-1.0 #(num_gates, 3) -> C is the actual selector ranging from [-1,1] based on learnable params
        if not self.training: #crisp choices for evaluation mode
            C = torch.where(C < -self.crisp_cutoff, -1.0,torch.where(C>self.crisp_cutoff,1.0, 0.0))
        C = C.unsqueeze(0).expand(bs,-1,-1) #(bs,num_gates,3)
        C_term1 = 1-torch.abs(C)
        C_term2 = torch.abs(C)
        inputs = C_term1 *inputs + (C+C_term2)/2 # every input now in range [0,1]
        if self.verb: print("#2. Compute intermediate input based on learnable selector -> inputs:\n", inputs)

        # 3. Unsqueeze
        inputs = inputs.unsqueeze(2)  # (bs, num_gates, 1, 3)
        if self.verb: print("#3. Unsqueeze -> inputs:\n", inputs)

        # 4. All permutations of 3 inputs: 3!=6 permutations
        perm_indices = torch.tensor(list(itertools.permutations(range(3))) ,device=inputs.device)
        inputs = inputs.expand(-1, -1, 6, -1)
        inputs = inputs.gather(3,
                               perm_indices.unsqueeze(0).unsqueeze(0).expand(inputs.shape[0],
                                                                             inputs.shape[1], -1, -1)) # (bs, num_gates, 6, 3)
        inputs = inputs
        if self.verb :print("#4. Permutations -> inputs:\n", inputs)
        # Input matrix so far for each sample, for each output neuron:
        #   u   x1 x2
        # 1: a   b  c
        # 2: a   c  b
        # 3: b   a  c
        # 4: b   c  a
        # 5: c   a  b
        # 6: c   b  a

        # 5. Interaction terms ux1 and ux2
        col4 = inputs[..., 0] * inputs[..., 1]
        col5 = inputs[..., 0] * inputs[..., 2]
        inputs = torch.cat([inputs, col4.unsqueeze(-1), col5.unsqueeze(-1)], dim=-1) # (bs,num_gates,6,5)
        if self.verb: print("#5. Add interactions -> inputs:\n", inputs)

        # 6. Transpose
        inputs = inputs.transpose(2, 3)             # (bs,num_gates,5,6)
        if self.verb :print("#6. Transpose -> inputs:\n", inputs)

        # 7. Matrix multiplication: Output that each permutation would yield on its own
        Signals = torch.matmul(self.M, inputs)      # (bs, num_gates, 3, 6)
        if self.verb: print("#7. Compute Signals -> Signals:\n", Signals)

        # 8. Apply learnable weights
        if self.training:
            wgts_prob = F.softmax(self.wgts, dim=1)
        else:
            wgts_prob = torch.zeros_like(self.wgts)
            wgts_prob.scatter_(1 ,self.wgts.argmax(-1 ,keepdim=True) ,1.0)

        wgts_exp = wgts_prob.unsqueeze(0).expand(Signals.shape[0], -1, -1)
        wgts_exp = wgts_exp.unsqueeze(-1)
        output = torch.matmul(Signals, wgts_exp)    # (bs,num_gates,3,1)
        if self.verb: print("#8. Weighted sum -> output:\n", output)

        # 9. Remove residual output (SKIPPED: In this FredkinLayer, there is no residual output to remove)
        output = output[:, :, 2, :]                # (bs,num_gates,3,1)
        #if self.verb: print("#9. Drop residual s -> output:\n", output)

        # 10. Flatten
        output = output.reshape(output.shape[0], -1) # (bs,dout=num_gates*1)
        if self.verb: print("#10. Flatten -> output:\n", output)

        return output







