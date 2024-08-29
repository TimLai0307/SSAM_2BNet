from .network import build_network

# build the  model
def build_model(args, training=False):
    return build_network(args, training)