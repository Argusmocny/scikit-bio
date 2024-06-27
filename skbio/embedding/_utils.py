# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

import numpy as np

def max_value_over_line_old(arr: np.ndarray, ystart: int, ystop: int,
						xstart: int, xstop: int) -> float:
	'''
	fix max value in row or column. When xstart == xstop - max
	value is calculated over other dimension
	and vice versa
	Args:
		arr (np.ndarray):
		ystart (int):
		ystop (int):
		xstart (int):
		xstop (int):
	Returns:
		float:
	'''
	max_value: float = 0
	if xstart == xstop:
		# iterate over array y array (1st dimension) slice
		max_value = arr[ystart, xstart]
		for yidx in range(ystart+1, ystop):
			if max_value > arr[yidx, xstart]:
				max_value = arr[yidx, xstart]
	else:
		max_value = arr[ystart, xstart]
		for xidx in range(xstart+1, xstop):
			if max_value > arr[ystart, xidx]:
				max_value = arr[ystart, xidx]
	return max_value


def fill_scorematrix_local(a: np.ndarray, gap_penalty: float = 0.0):
	'''
	fill score matrix with Smith-Waterman fashion

	Args:
		a: (np.array) 2D substitution matrix
		gap_penalty: (float)
	Return:
		b: (np.array)
	'''
	nrows: int = a.shape[0] + 1
	ncols: int = a.shape[1] + 1
	H: np.ndarray = np.zeros((nrows, ncols), dtype=np.float32)
	h_tmp: np.ndarray = np.zeros(4, dtype=np.float32)
	for i in range(1, nrows):
		for j in range(1, ncols):
			# no gap penalty for diagonal move
			h_tmp[0] = H[i-1, j-1] + a[i-1, j-1]
			# max over first dimension - y
			# max_{k >= 1} H_{i-k, j}
			#h_tmp[1] = max_value_over_line_gaps(H, 1, i+1, j, j, gap_pentalty=gap_penalty)
			h_tmp[1] = max_value_over_line_old(H, 1, i+1, j, j) - gap_penalty
			# max over second dimension - x
			h_tmp[2] = max_value_over_line_old(H, i, i, 1, j+1) - gap_penalty
			H[i, j] = np.max(h_tmp)
	return H


def fill_matrix_global(a: np.ndarray, gap_penalty: float):
	'''
	fill score matrix in Needleman-Wunch procedure - global alignment
	Params:
		a: (np.array)
		gap_penalty (float)
	Return:
		b: (np.array)
	'''
	nrows: int = a.shape[0] + 1
	ncols: int = a.shape[1] + 1
	H: np.ndarray = np.zeros((nrows, ncols), dtype=np.float32)
	h_tmp: np.ndarray = np.zeros(4, dtype=np.float32)
	for i in range(1, nrows):
		for j in range(1, ncols):
			# gap = abs(i - j)*gap_penalty
			h_tmp[0] = H[i-1, j-1] + a[i-1, j-1]
			h_tmp[1] = H[i-1, j] - gap_penalty
			h_tmp[2] = H[i, j-1] - gap_penalty
			H[i, j] = np.max(h_tmp)
	return H


def embedding_local_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
	'''
	compute X, Y similarity by matrix multiplication
	result shape [num X residues, num Y residues]
	Args:
		X, Y: - (np.ndarray 2D) protein embeddings as 2D tensors
		  [num residues, embedding size]
	Returns:
		density (np.ndarray)
	'''
	assert X.ndim == 2 and Y.ndim == 2
	assert X.shape[1] == Y.shape[1]

	xlen: int = X.shape[0]
	ylen: int = Y.shape[0]
	embdim: int = X.shape[1]
	# normalize
	emb1_norm: np.ndarray = np.empty((xlen, 1), dtype=np.float32)
	emb2_norm: np.ndarray = np.empty((ylen, 1), dtype=np.float32)
	emb1_normed: np.ndarray = np.empty((xlen, embdim), dtype=np.float32)
	emb2_normed: np.ndarray = np.empty((ylen, embdim), dtype=np.float32)
	density: np.ndarray = np.empty((xlen, ylen), dtype=np.float32)
	# numba does not support sum() args other then first
	emb1_norm = np.expand_dims(np.sqrt(np.power(X, 2).sum(1)), 1)
	emb2_norm = np.expand_dims(np.sqrt(np.power(Y, 2).sum(1)), 1)
	emb1_normed = X / emb1_norm
	emb2_normed = Y / emb2_norm
	density = (emb1_normed @ emb2_normed.T).T
	return density


def fill_score_matrix(sub_matrix: np.ndarray,
					  gap_penalty: Union[int, float] = 0.0,
					  globalmode: bool = False) -> np.ndarray:
	'''
	use substitution matrix to create score matrix
	set mode = local for Smith-Waterman like procedure (many local alignments)
	and mode = global for Needleamn-Wunsch like procedure (one global alignment)
	Params:
		sub_matrix: (np.array) substitution matrix in form of 2d
			array with shape: [num_res1, num_res2]
		gap_penalty: (float)
		mode: (str) set global or local alignment procedure
	Return:
		score_matrix: (np.array)
	'''
	assert gap_penalty >= 0, 'gap penalty must be positive'
	assert isinstance(globalmode, bool)
	assert isinstance(gap_penalty, (int, float, np.float32))
	assert isinstance(sub_matrix, np.ndarray), \
		'substitution matrix must be numpy array'
	# func fill_matrix require np.float32 array as input
	if not np.issubsctype(sub_matrix, np.float32):
		sub_matrix = sub_matrix.astype(np.float32)
	if not globalmode:
		score_matrix = fill_scorematrix_local(sub_matrix, gap_penalty=gap_penalty)
	else:
		score_matrix = fill_matrix_global(sub_matrix, gap_penalty=gap_penalty)
	return score_matrix

