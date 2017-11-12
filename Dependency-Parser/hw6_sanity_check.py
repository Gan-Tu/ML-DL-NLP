import numpy as np
import csv
from imp import reload
import hw6

test_id = 0

def check_result(test_name, expected, result):
    global test_id
    try:
        status = "Failed"
        if result == expected:
            status = "Passed"
        print("test #{}. checking {}: {}".format(
            test_id, test_name, status))
        test_id += 1
    except Exception as e:
        print("An error has occured")
        print(e)


def check_projective():
    non_proj_sent = [(1, "A", "DT", 2, "det"),
                     (2, "hearing", "NN", 4, "nsubj:pass"),
                     (3, "is", "VBZ", 4, "aux:pass"),
                     (4, "scheduled", "VBN", 0, "root"),
                     (5, "on", "IN", 7, "case"),
                     (6, "the", "DT", 7, "det"),
                     (7, "issue", "NN", 2, "nmod"),
                     (8, "today", "NN", 4, "nmod:tmod")]

    proj_sent = [(1, "You", "PRP", 2, "nsubj"),
                 (2, "wonder", "VBP", 0, "root"),
                 (3, "if", "IN", 6, "mark"),
                 (4, "he", "PRP", 6, "nsubj"),
                 (5, "was", "VBD", 6, "aux"),
                 (6, "manipulating", "VBG", 2, "ccomp"),
                 (7, "the", "DT", 8, "det"),
                 (8, "market", "NN", 6, "obj"),
                 (9, "with",	"IN", 12, "case"),
                 (10, "his", "PRP$", 12, "nmod:poss"),
                 (11, "bombing", "NN", 12, "compound"),
                 (12, "targets", "NNS", 6, "obl")]

    check_result('non-projective tree', False, is_projective(non_proj_sent))
    check_result('projective tree', True, is_projective(proj_sent))


def check_shift_first():
    wbuffer, stack, arcs, config, tran = [5, 4, 3, 2, 1], [0], [], [], []
    expected = [[5, 4, 3, 2], [0, 1], [], [([5, 4, 3, 2, 1], [0], [])], ['SHIFT']]
    result = [None, None, None, None, None]

    try:
        perform_shift(wbuffer, stack, arcs,config, tran)
    except Exception as e:
        print("Calling method {} failed".format(perform_shift.__name__))
        print(e)

    result = [wbuffer, stack, arcs, config, tran]
    check_result('first shift operation', expected, result)


def check_shift_regular():
    wbuffer, stack, arcs, config, tran = [5], [0, 1],\
        [('flat', 2, 3), ('flat', 2, 4), ('obj', 1, 2)],\
        [([5], [0, 1, 2], [('flat', 2, 3), ('flat', 2, 4)])],\
        ['RIGHTARC_obj']

    expected = [[], \
        [0, 1, 5], \
        [('flat', 2, 3), ('flat', 2, 4), ('obj', 1, 2)],\
        [([5], [0, 1, 2], [('flat', 2, 3), ('flat', 2, 4)]),\
        ([5], [0, 1], [('flat', 2, 3), ('flat', 2, 4), ('obj', 1, 2)])],\
        ['RIGHTARC_obj', 'SHIFT']]
    result = [None, None, None, None, None]

    try:
        perform_shift(wbuffer, stack, arcs, config, tran)
    except Exception as e:
        print("Calling method {} failed".format(perform_shift.__name__))
        print(e)

    result = [wbuffer, stack, arcs, config, tran]
    check_result('regular shift operation', expected, result)


def check_leftarc():
    direction, dep_label = 'LEFT', 'det'
    wbuffer, stack, arcs = [5, 4, 3], [0, 1, 2], []
    config, tran = [([5, 4, 3, 2], [0, 1], [])], ['SHIFT']

    expected = [[5, 4, 3], [0, 2], [('det', 2, 1)], \
        [([5, 4, 3, 2], [0, 1], []), ([5, 4, 3], [0, 1, 2], [])], \
        ['SHIFT', 'LEFTARC_det']]
    result = [None, None, None, None, None]

    try:
        perform_arc(direction, dep_label, wbuffer, stack, arcs, config, tran)
    except Exception as e:
        print("Calling method {} failed".format(perform_arc.__name__))
        print(e)

    result = [wbuffer, stack, arcs, config, tran]
    check_result('left arc operation', expected, result)


def check_rightarc():
    direction, dep_label = 'RIGHT', 'flat'
    wbuffer, stack, arcs = [5], [0, 1, 2, 4], [('flat', 2, 3)]
    config, tran = [([5, 4], [0, 1, 2], [('flat', 2, 3)])], ['SHIFT']

    expected = [[5], [0, 1, 2], [('flat', 2, 3), ('flat', 2, 4)], \
        [([5, 4], [0, 1, 2], [('flat', 2, 3)]),\
        ([5], [0, 1, 2, 4], [('flat', 2, 3)])],\
        ['SHIFT', 'RIGHTARC_flat']]
    result = [None, None, None, None, None]

    try:
        perform_arc(direction, dep_label, wbuffer, stack, arcs, config, tran)
    except Exception as e:
        print("Calling method {} failed".format(perform_arc.__name__))
        print(e)

    result = [wbuffer, stack, arcs, config, tran]
    check_result('right arc operation', expected, result)


def check_tree_to_actions_easy():
    wbuffer, stack, arcs, deps = [1], [0], [], {0: {(0, 1): 'root'}}
    expected = [[([1], [0], []), ([], [0, 1], [])], ['SHIFT', 'RIGHTARC_root']]
    config, tran = None, None
    result = [config, tran]

    try:
        config, tran = tree_to_actions(wbuffer, stack, arcs, deps)
    except Exception as e:
        print("Calling method {} failed".format(tree_to_actions.__name__))
        print(e)

    result = [config, tran]
    check_result('tree_to_actions (a)', expected, result)


def check_tree_to_actions_hard():
    wbuffer, stack, arcs = [4, 3, 2, 1], [0], []
    deps = {
        0: {(0, 1): 'root'}, \
        1: {(1, 2): 'punct', (1, 3): 'discourse', (1, 4): 'punct'}\
    }

    expected = [[
        ([4, 3, 2, 1], [0], []),\
        ([4, 3, 2], [0, 1], []), \
        ([4, 3], [0, 1, 2], []), \
        ([4, 3], [0, 1], [('punct', 1, 2)]), \
        ([4], [0, 1, 3], [('punct', 1, 2)]), \
        ([4], [0, 1], [('punct', 1, 2), ('discourse', 1, 3)]), \
        ([], [0, 1, 4], [('punct', 1, 2), ('discourse', 1, 3)]), \
        ([], [0, 1], [('punct', 1, 2), ('discourse', 1, 3), ('punct', 1, 4)])],
        ['SHIFT', 'SHIFT', 'RIGHTARC_punct', \
         'SHIFT', 'RIGHTARC_discourse', 'SHIFT', \
         'RIGHTARC_punct', 'RIGHTARC_root']]
    config, tran = None, None
    result = [config, tran]

    try:
        config, tran = tree_to_actions(wbuffer, stack, arcs, deps)
    except Exception as e:
        print("Calling method {} failed".format(tree_to_actions.__name__))
        print(e)

    result = [config, tran]
    check_result('tree_to_actions (b)', expected, result)


def check_action_to_tree_illegal():
    tree, predictions, wbuffer, stack, arcs = {}, \
        np.asarray([[0.5, 0.2, 0.2, 0.1]]), [], [0, 1, 2, 3, 4, 5], []

    expected = [{5: (4, 'punct')}, [], [0, 1, 2, 3, 4]]
    result = [None, None, None]

    try:
        action_to_tree(tree, predictions, wbuffer, stack, arcs)
    except Exception as e:
        print("Calling method {} failed".format(action_to_tree.__name__))
        print(e)

    result = [tree, wbuffer, stack]
    check_result('action_to_tree (illegal)', expected, result)


def check_action_to_tree_shift():
    tree, predictions, wbuffer, stack, arcs = {}, \
        np.asarray([[0.5, 0.2, 0.2, 0.1]]), [5, 4], [0, 1, 2, 3], []

    expected = [{}, [5], [0, 1, 2, 3, 4]]
    result = [None, None, None]

    try:
        action_to_tree(tree, predictions, wbuffer, stack, arcs)
    except Exception as e:
        print("Calling method {} failed".format(action_to_tree.__name__))
        print(e)

    result = [tree, wbuffer, stack]
    check_result('action_to_tree (shift)', expected, result)


def check_action_to_tree_leftarc():
    tree, predictions, wbuffer, stack, arcs = {}, \
        np.asarray([[0.1, 0.2, 0.1, 0.6]]), [5, 4], [0, 1, 2, 3], []

    expected = [{2: (3, 'det')}, [5, 4], [0, 1, 3]]
    result = [None, None, None]

    try:
        action_to_tree(tree, predictions, wbuffer, stack, arcs)
    except Exception as e:
        print("Calling method {} failed".format(action_to_tree.__name__))
        print(e)

    result = [tree, wbuffer, stack]
    check_result('action_to_tree (left arc)', expected, result)


def check_action_to_tree_rightarc():
    tree, predictions, wbuffer, stack, arcs = {}, \
        np.asarray([[0.3, 0.4, 0.2, 0.1]]), [5, 4], [0, 1, 2, 3], []

    expected = [{3: (2, 'punct')}, [5, 4], [0, 1, 2]]
    result = [None, None, None]

    try:
        action_to_tree(tree, predictions, wbuffer, stack, arcs)
    except Exception as e:
        print("Calling method {} failed".format(action_to_tree.__name__))
        print(e)

    result = [tree, wbuffer, stack]
    check_result('action_to_tree (right arc)', expected, result)


def check_all():
    result = np.hstack((check_projective(),
                       check_shift_first(),
                       check_shift_regular(),
                       check_leftarc(),
                       check_rightarc(),
                       check_tree_to_actions_easy(),
                       check_tree_to_actions_hard(),
                       check_action_to_tree_illegal(),
                       check_action_to_tree_shift(),
                       check_action_to_tree_leftarc(),
                       check_action_to_tree_rightarc()))
    return list(result)

if __name__ == "__main__":
    reload(hw6)
    hw6.reverse_labels = ['SHIFT', 'RIGHTARC_punct', 'RIGHTARC_flat', 'LEFTARC_det']
    from hw6 import is_projective, perform_shift, perform_arc, \
        tree_to_actions, action_to_tree
    result = check_all()
