
import torch
import torch.nn.functional as F

class GaLoreProjector:
    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type='std', quant=False, group_size=-1, n_bit=8, cos_threshold=0.4, gamma_proj=2, queue_size=5):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.ortho_matrix_scales = None
        self.ortho_matrix_zeros = None
        self.ortho_matrix_shape = None
        self.proj_type = proj_type

        # Quantization info for projector matrix
        self.quant = quant
        self.quant_group_size = group_size
        self.quant_n_bit = n_bit

        # Adaptive Update Subspace
        self.past_ortho_vector = None
        self.queue_size = queue_size
        self.queue = []
        self.cos_threshold = cos_threshold
        self.gamma_proj = gamma_proj
        self.svd_count = 0


    def project(self, full_rank_grad, iter):
        # TODO: This function only implementated with proj_type = 'std'
        assert self.proj_type == 'std'

        if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                float_ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                self.svd_count += 1
                if self.past_ortho_vector is not None:
                    if len(self.queue) == self.queue_size: self.queue.pop(0)
                    self.queue.append(F.cosine_similarity(self.past_ortho_vector, float_ortho_matrix[:1, :].clone().flatten(), dim=0).item())

                    if len(self.queue) == self.queue_size and sum(self.queue) / self.queue_size >= self.cos_threshold:
                        self.update_proj_gap = int(self.update_proj_gap * self.gamma_proj)
                self.past_ortho_vector = float_ortho_matrix[:1, :].clone().flatten()

                # Apply quantization to the projection matrix
                if self.quant:
                    self.ortho_matrix, self.ortho_matrix_scales, self.ortho_matrix_zeros, self.ortho_matrix_shape = self._quantize(float_ortho_matrix, q_group_size=self.quant_group_size, n_bit=self.quant_n_bit)
                else:
                    self.ortho_matrix = float_ortho_matrix

            if self.quant:
                float_ortho_matrix = self.ortho_matrix_scales * (self.ortho_matrix.to(self.ortho_matrix_scales.dtype) - self.ortho_matrix_zeros)
                float_ortho_matrix = float_ortho_matrix.reshape(self.ortho_matrix_shape)
            else:
                float_ortho_matrix = self.ortho_matrix

            # Project the gradient to the low rank subspace
            low_rank_grad = torch.matmul(full_rank_grad, float_ortho_matrix.t())

        else:
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:

                float_ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                self.svd_count += 1
                if self.past_ortho_vector is not None:
                    if len(self.queue) == self.queue_size: self.queue.pop(0)
                    self.queue.append(F.cosine_similarity(self.past_ortho_vector, float_ortho_matrix[:, :1].clone().flatten(), dim=0).item())

                    if len(self.queue) == self.queue_size and sum(self.queue) / self.queue_size >= self.cos_threshold:
                        self.update_proj_gap = int(self.update_proj_gap * self.gamma_proj)

                self.past_ortho_vector = float_ortho_matrix[:, :1].clone().flatten()

                # Apply quantization to the projection matrix
                if self.quant:
                    self.ortho_matrix, self.ortho_matrix_scales, self.ortho_matrix_zeros, self.ortho_matrix_shape = self._quantize(float_ortho_matrix, q_group_size=self.quant_group_size, n_bit=self.quant_n_bit)
                else:
                    self.ortho_matrix = float_ortho_matrix

            if self.quant:
                float_ortho_matrix = self.ortho_matrix_scales * (self.ortho_matrix.to(self.ortho_matrix_scales.dtype) - self.ortho_matrix_zeros)
                float_ortho_matrix = float_ortho_matrix.reshape(self.ortho_matrix_shape)
            else:
                float_ortho_matrix = self.ortho_matrix

            # Project the gradient to the low rank subspace
            low_rank_grad = torch.matmul(float_ortho_matrix.t(), full_rank_grad)

        return low_rank_grad

    def project_back(self, low_rank_grad):

        if self.proj_type == 'std':
            if self.quant:
                float_ortho_matrix = self.ortho_matrix_scales * (self.ortho_matrix.to(self.ortho_matrix_scales.dtype) - self.ortho_matrix_zeros)
                float_ortho_matrix = float_ortho_matrix.reshape(self.ortho_matrix_shape)
            else:
                float_ortho_matrix = self.ortho_matrix

            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, float_ortho_matrix)
            else:
                full_rank_grad = torch.matmul(float_ortho_matrix, low_rank_grad)

        return full_rank_grad * self.scale

    # svd decomposition
    def get_orthogonal_matrix(self, weights, rank, type):
        module_params = weights

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data
            
        U, s, Vh = torch.linalg.svd(matrix, full_matrices = False)
        
        #make the smaller matrix always to be orthogonal matrix
        if type=='right':
            A = U[:, :rank] @ torch.diag(s[:rank])
            B = Vh[:rank, :]
            
            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        elif type=='left':
            A = U[:, :rank]
            B = torch.diag(s[:rank]) @ Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        elif type=='full':
            A = U[:, :rank]
            B = Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B]
        else:
            raise ValueError('type should be left, right or full')

    def _quantize(self, w, q_group_size=-1, n_bit=8):
        org_w_shape = w.shape
        if q_group_size > 0:
            assert w.nelement() % q_group_size == 0
            w = w.reshape(-1, q_group_size)

        assert w.dim() == 2

        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        w = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)

        return w, scales, zeros, org_w_shape