import torch
import torch.nn.functional as F
from itertools import permutations

#-----------------------------------------
#CREATING DATASETS OF DIFFERENT COMPLEXITY
#-----------------------------------------


#1. BOOLEAN FUNCTIONS
def generate_bool_fct_data(name="and",return_triple=True,c=1.0,verb=False):
    if c=='a': temp = [0.0,0.0,1.0,1.0]
    elif c=='b': temp = [0.0,1.0,0.0,1.0]
    else: temp = [c,c,c,c]
    if return_triple:
        x_train = torch.tensor([[0.0, 0.0, temp[0]], [0.0, 1.0, temp[1]], [1.0, 0.0, temp[2]], [1.0, 1.0, temp[3]]])
    else:
        x_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

    if name=="and":
        labels_train = torch.tensor([0.0,0.0,0.0,1.0])
    elif name=="or":
        labels_train = torch.tensor([0.0,1.0,1.0,1.0])
    elif name=="implication":
        labels_train = torch.tensor([1.0,1.0,0.0,1.0])
    elif name=="biimplication":
        labels_train = torch.tensor([1.0,0.0,0.0,1.0])
    elif name=="xor":
        labels_train = torch.tensor([0.0,1.0,1.0,0.0])
    elif name=="nand":
        labels_train = torch.tensor([1.0,1.0,1.0,0.0])
    elif name=="backimplication":
        labels_train = torch.tensor([1.0,0.0,1.0,1.0])
    elif name=="true":
        labels_train = torch.tensor([1.0,1.0,1.0,1.0])
    elif name=="false":
        labels_train = torch.tensor([0.0,0.0,0.0,0.0])
    elif name=="notimplication":
        labels_train = torch.tensor([0.0,0.0,1.0,0.0])
    elif name=="notbackimplication":
        labels_train = torch.tensor([0.0,1.0,0.0,0.0])
    elif name=='a':
        labels_train = torch.tensor([0.0,0.0,1.0,1.0])
    elif name=='b':
        labels_train = torch.tensor([0.0,1.0,0.0,1.0])
    elif name=='notor':
        labels_train = torch.tensor([1.0,0.0,0.0,0.0])
    elif name=='nota':
        labels_train = torch.tensor([1.0,1.0,0.0,0.0])
    elif name=='notb':
        labels_train = torch.tensor([1.0,0.0,1.0,0.0])
    else:
        raise ValueError
    if verb: print("Boolean Function: ",x_train ," Labels: ", labels_train)
    return x_train,labels_train

#2. Function for defining easy data set of 0s and 1s
def create_majority_dataset(samplesize = 1000 , n_bits = 9 , p_one=0.5, device='cpu' ):
    x = torch.bernoulli(torch.full((samplesize,n_bits) , p_one , device=device)).float()
    labels = ( x.sum(dim=1) > (n_bits/2) ).float()
    return x,labels

def make_parity_dataset(n_bits: int, select_prop: float = 1.0, seed: int | None = None):
    if not (0.0 < select_prop <= 1.0):
        raise ValueError("select_prop must be in (0, 1].")

    # ----- 1. Enumerate all binary strings of length n_bits -----
    all_bits = list(itertools.product([0, 1], repeat=n_bits))
    total = len(all_bits)

    # ----- 2. Compute parity labels -----
    # parity = XOR of all bits, or equivalently sum % 2
    labels = [sum(bits) % 2 for bits in all_bits]

    # ----- 3. Randomly subsample -----
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    num_select = int(round(total * select_prop))
    idx = torch.randperm(total, generator=rng)[:num_select]

    # ----- 4. Convert to tensors -----
    X = torch.tensor([all_bits[i] for i in idx], dtype=torch.float32)
    y = torch.tensor([labels[i]   for i in idx], dtype=torch.float32)
    return X, y

def create_permutation_lookup_table(inputs):
    return {i: list(p) for i, p in enumerate(permutations(inputs))}


def print_fredkin_layer(module, lookup_table=None):
    """
    Print a Fredkin-like layer diagnostics:
      - Top 3 permutation weights
      - Connections if available
      - Selector values (2*sigmoid-1)
      - Crisp choices based on module.crisp_cutoff
    """
    if not hasattr(module, "wgts") or not module.wgts.requires_grad:
        #print("Module has no learnable 'wgts' parameter.")
        return

    W = module.wgts.data
    dout, din = W.shape
    print(f"{module._get_name()}  Top 3 Weights:")

    # Compute probabilities and top-k
    probs = F.softmax(W, dim=1)
    vals, idxs = torch.topk(probs, k=3, dim=1)

    # Check optional attributes
    has_conn = hasattr(module, "connections")
    has_sel  = hasattr(module, "selector")
    if has_sel:
        selectors = module.selector.detach().cpu()

    row_width = max(len(str(dout)), 2)
    col_width = len(str(din))

    for i in range(dout):
        print(f"Gate {i+1:<{row_width}} :")

        # Connections
        if has_conn:
            conn_i = module.connections[i].tolist()
            print(f"  Connections: {conn_i}")

        # Selectors and Crisp Choices
        if has_sel:
            #sel_vals = (2 * torch.sigmoid(selectors[i]) - 1).tolist()
            sel_vals = (selectors[i]/(1+torch.abs(selectors[i]))).tolist()
            crisp_vals = []
            for j, v in enumerate(sel_vals):
                if v < -module.crisp_cutoff:
                    crisp_vals.append("0")
                elif v > module.crisp_cutoff:
                    crisp_vals.append("1")
                else:
                    crisp_vals.append(chr(65 + j))  # "A", "B", "C"

            sel_str = "  Selectors: " + ", ".join(
                f"{chr(65+j)} = {v:.4f}" for j, v in enumerate(sel_vals)
            )
            crisp_str = " , Crisp Choices: " + ", ".join(
                f"{chr(65+j)}={cv}" for j, cv in enumerate(crisp_vals)
            )
            print(sel_str + crisp_str)

        # Top-3 permutation weights
        perm_parts = []
        for j, v in zip(idxs[i], vals[i]):
            if lookup_table is not None:
                perm_str = lookup_table[j.item()]
            else:
                perm_str = str(j.item())
            perm_parts.append(
                f"Index {j.item():<{col_width}} ({perm_str}): {v.item()*100:5.1f}%"
            )
        print("  Permutation weights: " + ", ".join(perm_parts))


def print_fredkin_layer3plus6(module, lookup_table=None):
    """
    Condensed Fredkin-like layer diagnostics:
    Shows port assignments for each gate after applying
    connections, selectors, crisp cutoff, and best permutation.
    """
    if not hasattr(module, "wgts") or not module.wgts.requires_grad:
        return

    W = module.wgts.data
    dout, din = W.shape
    print(f"{module._get_name()}  Port Assignment:")

    # Probabilities
    probs = F.softmax(W, dim=1)
    vals, idxs = torch.topk(probs, k=1, dim=1)  # just best permutation

    has_conn = hasattr(module, "connections")
    has_sel  = hasattr(module, "selector")

    if has_sel:
        selectors = module.selector.detach().cpu()

    for i in range(dout):
        # Step 1: take connection indices → map to characters
        if has_conn:
            conn_i = module.connections[i].tolist()
            conn_labels = [str(j) for j in conn_i] # "a","b","c",...
        else:
            conn_labels = [f"in{j}" for j in range(din)]

        # Step 2: apply crisp cutoff to selectors
        if has_sel:
            #sel_vals = (2 * torch.sigmoid(selectors[i]) - 1).tolist()
            sel_vals = (selectors[i] / (1 + torch.abs(selectors[i]))).tolist()
            for j, v in enumerate(sel_vals):
                if v < -module.crisp_cutoff:
                    conn_labels[j] = "0(const.)"
                elif v > module.crisp_cutoff:
                    conn_labels[j] = "1(const.)"
                else:
                    conn_labels[j] = f"in{conn_i[j]}"

        # Step 3: apply best permutation
        best_perm_idx = idxs[i][0].item()
        if lookup_table is not None:
            perm = lookup_table[best_perm_idx]
        else:
            perm = list(range(din))

        # Reorder connections according to permutation
        permuted_outputs = [conn_labels[j] for j in perm]

        # Step 4: Print result
        print(f"Gate {i+1:<2}: " +
              f"u={permuted_outputs[0]:<10}, x1={permuted_outputs[1]:<10}, x2={permuted_outputs[2]:<10}")
