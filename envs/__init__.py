
def get_environment(hps, split):
    if hps.env == 'img':
        from envs.img import Env
        return Env(hps, split)
    elif hps.env == 'img_term':
        from envs.img_term import Env
        return Env(hps, split)
    elif hps.env == 'vec':
        from envs.vec import Env
        return Env(hps, split)
    elif hps.env == 'vec_term':
        from envs.vec_term import Env
        return Env(hps, split)
    else:
        raise ValueError()