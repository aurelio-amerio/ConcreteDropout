def get_weight_regularizer(N, l=1e-2, tau=0.1):
    return l**2 / (tau * N)


def get_dropout_regularizer(N, tau=0.1, cross_entropy_loss=False):
    reg = 1 / (tau * N)
    if not cross_entropy_loss:
        reg *= 2
    return reg