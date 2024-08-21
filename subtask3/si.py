import torch

def update_omega(_model, _device, W, epsilon):
    """
    after completing training on a task, update the per-parameter 
    regularization strength (omega)
    [W]         <dict> estimated parameter-specific contribution 
                       to changes in total loss of completed task
    [epsilon]   <float> dampening parameter 
                        (to bound [omega] when [p_change] goes to 0)    
    """
    # Loop over all parameters
    for n, p in _model.named_parameters():
        if p.requires_grad:
            n = n.replace('.', '__')
            W_new = W[n].to(_device)

            # Find/calculate new values for quadratic penalty on parameters
            p_prev = getattr(_model, '{}_SI_prev_task'.format(n))
            p_current = p.detach().clone()
            p_change = p_current - p_prev
            omega_add = W_new/(p_change**2 + epsilon)
            try:
                omega = getattr(_model, '{}_SI_omega'.format(n))
            except AttributeError:
                omega = p.detach().clone().zero_()
            omega_new = omega + omega_add

            # Store these new values in the model
            _model.register_buffer('{}_SI_prev_task'.format(n), p_current)
            _model.register_buffer('{}_SI_omega'.format(n), omega_new)


def surrogate_loss(_model):
    """
    calculate SI's surrogate loss
    """

    try:
        losses = []
        for n, p in _model.named_parameters():
            if p.requires_grad:
                # Retrieve previous parameter values and 
                # their normalized path integral (i.e., omega)
                n = n.replace('.', '__')
                prev_values = getattr(_model, '{}_SI_prev_task'.format(n))
                omega = getattr(_model, '{}_SI_omega'.format(n))
                # Calculate SI's surrogate loss, sum over all parameters
                losses.append((omega * (p-prev_values)**2).sum())
        return sum(losses)
    except AttributeError:
        # SI-loss is 0 if there is no stored omega yet
        return torch.tensor(0., device=next(iter(_model.parameters())).device)
