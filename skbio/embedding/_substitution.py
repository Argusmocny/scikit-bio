# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

from typing import Union

import numpy as np
from skbio.stats.distance import DissimilarityMatrix
from skbio.embedding import ProteinEmbedding
from skbio.embedding._utils import fill_score_matrix, embedding_local_similarity


class EmbeddingSubstitionMatrix(DissimilarityMatrix):
    """
    The substitution matrix is constructed from residue-normalized embeddings for both proteins.
    This substitution matrix can be further used as an input for Needleman-Wunsch alignment
    or further transformed into a scoring matrix for Smith-Waterman local alignments,
    providing both local and global alignments in the output.

    This class store both substitution matrix and scoring matrix
    """
    def __init__(self, substitution_matrix: np.ndarray, mode: str = 'local'):
        self.scores = fill_score_matrix(substitution_matrix, globalmode=False if mode == 'local' else True)

    @classmethod
    def from_embedding(cls, emb1: ProteinEmbedding, emb2: ProteinEmbedding, mode='local'):        
        
        substitution_matrix = embedding_local_similarity(emb1.embedding, emb2.embedding)
        return cls(substitution_matrix, mode)

