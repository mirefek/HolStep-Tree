import numpy as np
import tensorflow as tf
import test_tree_utils as ttu
from generator_network import Network
from tf_tree_utils import TreePlaceholder

import sys
import traceback_utils

if __name__ == "__main__":

    sys.excepthook = traceback_utils.shadow('/usr/')

    lines = []
    lines.append("P * f1 f2\n")
    lines.append("P * / b0 * ! * * c= * * cGSPEC / b1 * b0 * cSETSPEC b2 b1 * b0 / b1 / b2 * * c/\ b2 * * c= b1 b3 f0\n")
    lines.append("P * * * * * f1 f2 f3 f4 f5 f100\n")
    lines.append("P * * c= * * c- * cSUC f0 * cSUC f1 * * c- f0 f1\n")
    lines.append("P * * c= / b0 * f0 b0 f0\n")
    lines.append("P / b0 * b1 b2\n")
    lines.append("P cT\n")
    #lines.append("P * * c==> * c~ * ccollinear * * cINSERT f0 * * cINSERT f1 * * cINSERT f2 cEMPTY * * c==> * c~ * * * cbarV f3 * cNUMERAL * cBIT1 * cBIT1 c_0 * * ccc_uh f3 * * cCONS f0 * * cCONS f1 * * cCONS * f4 f5 cNIL * * c==> * cpacking f3 * * c==> * csaturated f3 * * c==> * * c= * * cmcell4 f3 * * ccc_uh f3 * * cCONS f0 * * cCONS f1 * * cCONS * f4 f5 cNIL * * ccc_cell f3 * * cCONS f0 * * cCONS f1 * * cCONS * f4 f5 cNIL * * c==> * * c= f6 f5 * * c==> * * c= * * ccc_ke f3 * * cCONS f0 * * cCONS f1 * * cCONS * f4 f6 cNIL * cNUMERAL * cBIT0 * cBIT0 * cBIT1 c_0 * * c==> * * c= * * cEL * cNUMERAL c_0 * * ccc_uh f3 * * cCONS f0 * * cCONS f1 * * cCONS * f4 f6 cNIL f0 * * c==> * * c= * * cEL * cNUMERAL * cBIT0 * cBIT1 c_0 * * ccc_uh f3 * * cCONS f0 * * cCONS f1 * * cCONS * f4 f6 cNIL * f4 f6 * * c==> * * c= * * cEL * cNUMERAL * cBIT1 c_0 * * ccc_uh f3 * * cCONS f0 * * cCONS f1 * * cCONS * f4 f6 cNIL f1 * * c==> * * c< * cNUMERAL * cBIT1 c_0 f7 * * c==> * * cperiodic f4 f7 * * c==> * * cleaf f3 * * cCONS f0 * * cCONS f1 * * cCONS * f4 f6 cNIL * * c==> * * * * * cleaf_rank f3 * * cCONS f0 * * cCONS f1 cNIL f2 f7 f4 * * c==> * * * * * ccc_4 f3 f0 f1 f4 f6 * * c==> * ! / b0 * ! / b1 * * c= * * cIN b1 b0 * b0 b1 * * c==> * ! / b0 * ! / b1 * * c==> * * c/\ * cpacking b0 * * c/\ * csaturated b0 * * cleaf b0 b1 * * c/\ * * cIN * * ccc_uh b0 b1 * * cbarV b0 * cNUMERAL * cBIT1 * cBIT1 c_0 * * c/\ * * c= * * ctruncate_simplex * cNUMERAL * cBIT0 * cBIT1 c_0 * * ccc_uh b0 b1 b1 * * c= * * comega_list b0 * * ccc_uh b0 b1 * * ccc_pe1 b0 b1 cF")

    real_structure = ttu.lines_to_tree_structure(lines)

    print(ttu.vocabulary)
    network = Network(20, len(lines), ttu.vocabulary, ttu.reverse_voc, max_steps = 100)
    for i in range(401):

        types_loss_acc, const_loss_acc = network.train(real_structure, ttu.preselection, np.arange(len(lines)))
        if i%20 == 0:
            print("{}: types {},  const {}".format(i, types_loss_acc, const_loss_acc))
            predictions = network.predict(range(len(lines)))
            for ori, pred in zip(lines, predictions):
                print("Original: {}".format(ori.rstrip()))
                print("Prediction: {}".format(pred))
