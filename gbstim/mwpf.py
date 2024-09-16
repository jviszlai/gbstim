import pathlib
import numpy as np
from sinter import CompiledDecoder, Decoder
from stim import DetectorErrorModel
from scipy.sparse import csc_matrix
from beliefmatching import detector_error_model_to_check_matrices

from mwpf import HyperEdge, SolverInitializer, SolverSerialJointSingleHair, SyndromePattern

class CompiledMWPF(CompiledDecoder):

    def __init__(self, check_matrices, decoder):
        self.check_matrices = check_matrices
        self.decoder = decoder

    def decode_shots_bit_packed(self, 
                                bit_packed_detection_event_data: np.ndarray
                               ) -> np.ndarray:
        obs_flip_data = []
        for shot_data in bit_packed_detection_event_data:
            unpacked_data = np.unpackbits(shot_data, bitorder='little', count=self.check_matrices.check_matrix.shape[0])
            self.decoder.solve(SyndromePattern(unpacked_data))
            err_idx = self.decoder.subgraph()
            pred_errs = csc_matrix((np.ones(len(err_idx)), err_idx, [0, len(err_idx)]), shape=(len(err_idx), 1))
            self.decoder.clear()
            obs_pred = (self.check_matrices.observables_matrix @ pred_errs) % 2
            obs_flip_data.append(np.packbits(obs_pred, bitorder='little'))

        return np.array(obs_flip_data)

class MWPF(Decoder):

    def __init__(self, prob_scale: float=1e-3, **kwargs):
        self.prob_scale = prob_scale
        self.decoder_kwargs = kwargs
    
    def build_decoder_solver(self, 
                            check_matrices
                            ) -> SolverSerialJointSingleHair:
        num_dets = check_matrices.check_matrix.shape[0]
        num_errs = check_matrices.check_matrix.shape[1]
        weighted_edges = [
            HyperEdge(
                check_matrices.check_matrix.getcol(i).nonzero()[0],
                np.round(self.prob_scale * 
                    np.log((1 - check_matrices.priors[i])/check_matrices.priors[i])
                ).astype(int)
            )
            for i in range(num_errs)
        ]
        initializer = SolverInitializer(num_dets, weighted_edges)
        return SolverSerialJointSingleHair(initializer)
        

    def compile_decoder_for_dem(self, 
                                dem: DetectorErrorModel
                               ) -> CompiledDecoder:
        
        check_matrices = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
        decoder = self.build_decoder_solver(check_matrices)
        return CompiledMWPF(check_matrices, decoder)

    def decode_via_files(self, 
                         *, 
                         num_shots: int, 
                         num_dets: int, 
                         num_obs: int, 
                         dem_path: pathlib.Path, 
                         dets_b8_in_path: pathlib.Path, 
                         obs_predictions_b8_out_path: pathlib.Path, 
                         tmp_dir: pathlib.Path
                        ) -> None:
        raise NotImplementedError()