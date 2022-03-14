
def get_dose(config):
    if config.ood_scoring.network == 'dose':
        from .dose import DOSE
        return DOSE(config).to(config.device)
    elif config.ood_scoring.network == 'dose1d':
        from .dose import DOSE1d
        return DOSE1d(config).to(config.device)
