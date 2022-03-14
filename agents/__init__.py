
def get_agent(hps, env, model, detector):
    if hps.agent == 'img_cls_hppo':
        from agents.img_cls_hppo import PPOPolicy
        return PPOPolicy(hps, env, model, detector)
    elif hps.agent == 'img_cls_ppo':
        from agents.img_cls_ppo import PPOPolicy
        return PPOPolicy(hps, env, model, detector)
    elif hps.agent == 'img_cls_tppo':
        from agents.img_cls_tppo import PPOPolicy
        return PPOPolicy(hps, env, model, detector)
    elif hps.agent == 'img_cls_rand':
        from agents.img_cls_rand import RandPolicy
        return RandPolicy(hps, env, model, detector)
    elif hps.agent == 'img_cls_hgcppo':
        from agents.img_cls_hgcppo import PPOPolicy
        return PPOPolicy(hps, env, model, detector)
    elif hps.agent == 'img_cls_hgkppo':
        from agents.img_cls_hgkppo import PPOPolicy
        return PPOPolicy(hps, env, model, detector)
    elif hps.agent == 'vec_cls_hppo':
        from agents.vec_cls_hppo import PPOPolicy
        return PPOPolicy(hps, env, model, detector)
    elif hps.agent == 'vec_cls_ppo':
        from agents.vec_cls_ppo import PPOPolicy
        return PPOPolicy(hps, env, model, detector)
    elif hps.agent == 'vec_cls_gkppo':
        from agents.vec_cls_gkppo import PPOPolicy
        return PPOPolicy(hps, env, model, detector)
    elif hps.agent == 'img_rec_hppo':
        from agents.img_rec_hppo import PPOPolicy
        return PPOPolicy(hps, env, model, detector)
    elif hps.agent == 'img_rec_ppo':
        from agents.img_rec_ppo import PPOPolicy
        return PPOPolicy(hps, env, model, detector)
    elif hps.agent == 'img_rec_rand':
        from agents.img_rec_rand import RandPolicy
        return RandPolicy(hps, env, model, detector)
    elif hps.agent == 'img_rec_hgkppo':
        from agents.img_rec_hgkppo import PPOPolicy
        return PPOPolicy(hps, env, model, detector)
    else:
        raise ValueError()