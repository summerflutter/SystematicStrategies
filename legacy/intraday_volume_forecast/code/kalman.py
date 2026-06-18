import numpy as np

def specify_uniss(data: np.ndarray, volume_model: dict) -> dict:
    """
    Prepare UNISS object from log-volume matrix and a volume_model spec.
    data: 2D array (n_bin × n_day) of log-volume.
    volume_model: dict with keys 'par', 'init', and 'converged'.
    """
    # Flatten column-wise (R's unlist)
    y = data.flatten(order='F')
    n_bin, n_day = data.shape
    n_total = n_bin * n_day

    # Default EM initial values
    overall_mean = y.mean()
    init_default = {
        'x0': np.array([overall_mean, 0.0]),
        'a_eta': 1.0,
        'a_mu': 0.0,
        'r': 1e-4,
        'var_eta': 1e-4,
        'var_mu': 1e-4,
        'V0': np.array([1e-3, 1e-7, 1e-5]),
        'phi': data.mean(axis=1) - overall_mean
    }

    # Build uniss_obj
    uniss_obj = {
        'par': {},
        'y': y,
        'n_bin': n_bin,
        'n_day': n_day,
        'n_bin_total': n_total,
        'converged': volume_model['converged']
    }

    # Fill each parameter: if not converged, use init or default; else use fixed
    for name in ['a_eta','a_mu','var_eta','var_mu','r','phi','x0','V0']:
        if not volume_model['converged'][name]:
            if name in volume_model.get('init', {}):
                uniss_obj['par'][name] = np.array(volume_model['init'][name])
            else:
                uniss_obj['par'][name] = init_default[name]
        else:
            uniss_obj['par'][name] = np.array(volume_model['par'][name])

    return uniss_obj


def uniss_kalman(uniss_obj: dict, mode: str = "em_update") -> dict:
    """
    Perform one pass of Kalman filter, smoother, or EM update on uniss_obj.
    mode: "filter", "smoother", or "em_update".
    """
    y = uniss_obj['y']
    n = uniss_obj['n_bin_total']
    n_bin = uniss_obj['n_bin']
    n_day = uniss_obj['n_day']

    # Allocate arrays
    xtt1 = np.zeros((2, n))
    Vtt1 = np.zeros((2, 2, n))
    xtt  = np.zeros((2, n))
    Vtt  = np.zeros((2, 2, n))
    Kt   = np.zeros((2, n))

    # Unpack parameters
    p = uniss_obj['par']
    A_jump  = np.array([[p['a_eta'], 0], [0, p['a_mu']]])
    A_intra = np.array([[1.0, 0], [0, p['a_mu']]])
    Q_jump  = np.diag([p['var_eta'], p['var_mu']])
    Q_intra = np.diag([0.0,      p['var_mu']])
    r       = p['r']
    phi     = np.tile(p['phi'], n_day)
    C       = np.array([[1.0, 1.0]])

    # Initialization (t = 0)
    xtt1[:,0] = p['x0']
    V0 = np.array([[p['V0'][0], p['V0'][1]],
                   [p['V0'][1], p['V0'][2]]])
    Vtt1[:,:,0] = V0
    S0 = (C @ V0 @ C.T + r)[0,0]
    Kt[:,0] = (V0 @ C.T).flatten() / S0
    res0 = y[0] - phi[0] - (C @ xtt1[:,0])[0]
    xtt[:,0] = xtt1[:,0] + Kt[:,0] * res0
    Vtt[:,:,0] = V0 - np.outer(Kt[:,0], (C @ V0).flatten())

    # Kalman FILTER
    for i in range(n-1):
        # Predict
        if (i+1) % n_bin == 0:
            xtt1[:,i+1] = A_jump @ xtt[:,i]
            Vtt1[:,:,i+1] = A_jump @ Vtt[:,:,i] @ A_jump.T + Q_jump
        else:
            xtt1[:,i+1] = A_intra @ xtt[:,i]
            Vtt1[:,:,i+1] = A_intra @ Vtt[:,:,i] @ A_intra.T + Q_intra

        # Gain
        S = (C @ Vtt1[:,:,i+1] @ C.T + r)[0,0]
        Kt[:,i+1] = (Vtt1[:,:,i+1] @ C.T).flatten() / S

        # Update
        res = y[i+1] - phi[i+1] - (C @ xtt1[:,i+1])[0]
        xtt[:,i+1] = xtt1[:,i+1] + Kt[:,i+1] * res
        Vtt[:,:,i+1] = Vtt1[:,:,i+1] - np.outer(Kt[:,i+1], (C @ Vtt1[:,:,i+1]).flatten())

    result = {'xtt1': xtt1, 'Vtt1': Vtt1, 'Kt': Kt, 'xtt': xtt, 'Vtt': Vtt}
    if mode == "filter":
        return result

    # Kalman SMOOTHER
    xtT = np.zeros_like(xtt)
    VtT = np.zeros_like(Vtt)
    Lt  = np.zeros((2,2,n-1))

    xtT[:,-1] = xtt[:,-1]
    VtT[:,:,-1] = Vtt[:,:,-1]
    for i in range(n-2, -1, -1):
        A = A_jump if (i+1) % n_bin == 0 else A_intra
        P_pred = Vtt1[:,:,i+1]
        Lt[:,:,i] = Vtt[:,:,i] @ A.T @ np.linalg.inv(P_pred)
        xtT[:,i] = xtt[:,i] + Lt[:,:,i] @ (xtT[:,i+1] - xtt1[:,i+1])
        VtT[:,:,i] = Vtt[:,:,i] + Lt[:,:,i] @ (VtT[:,:,i+1] - P_pred) @ Lt[:,:,i].T

    result.update({'xtT': xtT, 'VtT': VtT, 'Lt': Lt,
                   'x0T': xtT[:,0], 'V0T': VtT[:,:,0]})
    if mode == "smoother":
        return result

    # EM UPDATE
    Pt   = np.zeros_like(Vtt)
    Ptt1 = np.zeros_like(Vtt)
    for i in range(n):
        Pt[:,:,i] = VtT[:,:,i] + np.outer(xtT[:,i], xtT[:,i])
    for i in range(1, n):
        Ptt1[:,:,i] = VtT[:,:,i] @ Lt[:,:,i-1].T + np.outer(xtT[:,i], xtT[:,i-1])

    # Indices for jump intervals (zero-based)
    jump_idx = [(k+1)*n_bin - 1 for k in range(n_day)]

    new_par = {k: v.copy() for k,v in uniss_obj['par'].items()}
    unfitted = [k for k,v in uniss_obj['converged'].items() if not v]

    for name in unfitted:
        if name == 'x0':
            new_par['x0'] = result['x0T']
        elif name == 'V0':
            V0T = result['V0T']
            new_par['V0'] = np.array([V0T[0,0], V0T[1,0], V0T[1,1]])
        elif name == 'phi':
            resid = y - (C @ xtT).flatten()
            mat = resid.reshape((n_bin, n_day), order='F')
            phi_est = mat.mean(axis=1)
            new_par['phi'] = phi_est - phi_est.mean()
        elif name == 'r':
            phi_mat = np.tile(new_par.get('phi', p['phi']), n_day)
            cv = (C @ xtT).flatten()
            term = (y**2 +
                    np.array([ (C @ Pt[:,:,i] @ C.T)[0,0] for i in range(n) ]) -
                    2*y*cv +
                    phi_mat**2 -
                    2*y*phi_mat +
                    2*phi_mat*cv)
            new_par['r'] = term.mean()
        elif name == 'a_eta':
            num = sum(Ptt1[0,0,j] for j in jump_idx)
            den = sum(Pt[0,0,j-1] for j in jump_idx)
            new_par['a_eta'] = num / den
        elif name == 'a_mu':
            new_par['a_mu'] = Ptt1[1,1,1:].sum() / Pt[1,1,:-1].sum()
        elif name == 'var_eta':
            a_eta = new_par['a_eta'] if 'a_eta' in unfitted else p['a_eta']
            vals = [Pt[0,0,j] + a_eta**2 * Pt[0,0,j-1] - 2*a_eta*Ptt1[0,0,j] for j in jump_idx]
            new_par['var_eta'] = np.mean(vals)
        elif name == 'var_mu':
            a_mu = new_par['a_mu'] if 'a_mu' in unfitted else p['a_mu']
            vals = [Pt[1,1,j] + a_mu**2 * Pt[1,1,j-1] - 2*a_mu*Ptt1[1,1,j] for j in range(1,n)]
            new_par['var_mu'] = np.mean(vals)

    result['new_par'] = new_par
    return result
