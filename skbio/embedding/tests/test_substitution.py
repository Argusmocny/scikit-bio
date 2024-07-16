# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import TestCase, main

import numpy as np
import numpy.testing as npt
import pandas as pd
from scipy.spatial.distance import euclidean

from skbio.embedding._protein import ProteinVector, ProteinEmbedding
from skbio.embedding._substitution import EmbeddingSubstitionMatrix

from skbio.embedding._embedding import (
    Embedding
)


class EmbeddingTests(TestCase):
    def setUp(self):
        self.emb_x = np.random.randn(62, 10)
        self.emb_y = np.random.random(101, 10)
        self.seq_x = "IGKEEIQQRLAQFVDHWKELKQLAAARGQRLEESLEYQQFVANVEEEEAWINEKMTLVASED"
        self.seq_y = "MQIQRIYTKDISFEAPNAPHVFQKDWGAIVNGINNSFTNRRDQKSLGELMKLIVKSFLRHPESGLPAPRKPYERLQFLLKEGQGTQDFELVAMNSEVRKAGLQYPFIV"

    def test_init(self):

        x = ProteinEmbedding(self.emb_x, self.seq_x)
        y = ProteinEmbedding(self.emb_y, self.seq_y)

        submx = EmbeddingSubstitionMatrix(x, y)

        self.assertEqual(submx.scores.shape[0], x.shape[0])
        self.assertEqual(submx.scores.shape[1], y.shape[1])

if __name__ == "__main__":
    main()
