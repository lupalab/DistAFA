
def get_model(args):
    if args.model == 'acflow':
        from models.acflow import ACFlow
        model = ACFlow(args)
    elif args.model == 'acflow_classifier':
        from models.acflow_classifier import FlowClassifier
        model = FlowClassifier(args)
    elif args.model == 'actan':
        from models.actan import ACFlow
        model = ACFlow(args)
    elif args.model == 'actan_classifier':
        from models.actan_classifier import FlowClassifier
        model = FlowClassifier(args)
    else:
        raise Exception()
    
    return model