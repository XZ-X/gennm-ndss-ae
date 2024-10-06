
import tree_sitter

def find_first_recursively_opt(node, type_name, depth=0):
    if node.type == type_name:
        return node
    if depth > 20:
        return None
    for child in node.named_children:
        result = find_first_recursively_opt(child, type_name, depth + 1)
        if result is not None:
            return result
    return None

def find_all_recursively(node, type_name, depth=0):
    result = []
    if node.type == type_name:
        result.append(node)
    if depth > 20:
        return result
    for child in node.named_children:
        result.extend(find_all_recursively(child, type_name, depth + 1))
    return result

def get_first_opt(node, type_name):
    for child in node.named_children:
        if child.type == type_name:
            return child
    return None

def get_all(node, type_name):
    result = []
    for child in node.named_children:
        if child.type == type_name:
            result.append(child)
    return result
