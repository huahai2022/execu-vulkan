def _filter_sym_size_users(node):
    return []


def _is_conv_node(node):
    return False


def _is_conv_transpose_node(node):
    return False


def get_new_attr_name_with_prefix(module, prefix):
    base = f"{prefix}_0"
    index = 0
    while hasattr(module, base):
        index += 1
        base = f"{prefix}_{index}"
    return base
