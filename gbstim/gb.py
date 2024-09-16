from gbstim.device import PhysicalQubit, DataQubit, MeasureQubit, Device
import numpy as np
import stim
import galois
import collections
from ldpc import bposd_decoder

class LogicalQubit():

    def __init__(self, x_obs, z_obs) -> None:
        self.x_obs = x_obs
        self.z_obs = z_obs

class GBMeasureQubit(MeasureQubit):

    def __init__(self, l_data, r_data, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.l_data = l_data
        self.r_data = r_data

class GBCode():

    def __init__(self, device: Device, A_poly: list[tuple[int, int]], B_poly: list[tuple[int, int]], l, m):
        self.device = device

        # Code Construction
        def permutation_matrix(dim, pow):
            I = galois.GF2.Identity(dim)
            S = np.delete(I, 0, 0)
            return np.linalg.matrix_power(np.vstack((S, I[0])), pow)

        self.M = [np.kron(permutation_matrix(l, i), galois.GF2.Identity(m)) @ 
                  np.kron(galois.GF2.Identity(l) , permutation_matrix(m, j)) 
                  for i in range(l) 
                  for j in range(m)]

        def mat2poly(mat: np.ndarray):
            poly = []
            for i, term in enumerate(self.M):
                overlap = [mat[i,j] == 1 for i, j in zip(*np.where(term == 1))]
                if np.alltrue(overlap):
                    poly.append(i)
            return poly
        
        def poly2mat(M_idx):
            mat = galois.GF2.Zeros(self.M[0].shape)
            for term_idx in M_idx:
                mat += self.M[term_idx]
            return mat

        def calculate_periodicity(i, x_pow, y_pow):
            if i < m * (l - x_pow) and (i % m) < (m - y_pow):
                return 0 # No periodicity
            elif i >= m * (l - x_pow) and (i % m) < (m - y_pow):
                return 1 # Only horizontal periodicity
            elif i < m * (l - x_pow) and (i % m) >= (m - y_pow):
                return 2 # Only vertical periodicity
            else:
                return 3 # Horizontal and vertical periodicity

        A_idx = [x * m + y for x,y in A_poly]
        B_idx = [x * m + y for x,y in B_poly]
        
        A = poly2mat(A_idx)
        A_terms = [poly2mat([poly]) for poly in A_idx]
        B = poly2mat(B_idx)
        B_terms = [poly2mat([poly]) for poly in B_idx]
        
        self.Gx = np.hstack((A, B))
        self.Gz = np.hstack((B.T, A.T))
        self.l = l
        self.m = m

        self.n = 2 * l * m
        self.k = self.n - np.linalg.matrix_rank(self.Gx) - np.linalg.matrix_rank(self.Gz)

        # Data Qubits
        self.data_1 = [DataQubit(f'd1@{x * m + y}', x * m + y, (2*x + 1, 2*y), (2*x + 1, 2*y)) for x in range(l) for y in range(m)]
        self.data_2 = [DataQubit(f'd2@{x * m + y}', len(self.data_1) + x * m + y, (2*x, 2*y + 1), (2*x, 2*y + 1)) for x in range(l) for y in range(m)]

        for data in self.data_1 + self.data_2:
            self.device.device_array[data.global_coords[1], data.global_coords[0]] = data
        
        # Logical Observables
        def get_ab(f, h):
            ab_idx = []
            a = [i for i in range(l*m)]
            b = [j for j in range(l*m)]
            for i in a:
                for j in b:
                    alpha = self.M[i]
                    beta = self.M[j]
                    if mat2poly(alpha.T @ beta)[0] in mat2poly(poly2mat(f) @ poly2mat(h)):
                        ab_idx.append((i,j))
                        break
                b.remove(j)
                if len(ab_idx) == (self.k // 2):
                    break
            return ab_idx
        
        def get_obs_mat(f, g, h, ab_idx):
            X = galois.GF2.Zeros((self.k, 2*l*m))
            Z = galois.GF2.Zeros((self.k, 2*l*m))
            idx = 0
            for a,b in ab_idx:
                alpha = self.M[a]
                beta = self.M[b]
                X[idx, :l*m] = (alpha @ poly2mat(f))[0]
                X[idx, l*m:] = galois.GF2.Zeros((1, l*m))
                Z[idx, :l*m] = (beta @ poly2mat(h).T)[0]
                Z[idx, l*m:] = (beta @ poly2mat(g).T)[0]
                idx += 1
                X[idx, :l*m] = (alpha @ poly2mat(g))[0]
                X[idx, l*m:] = (alpha @ poly2mat(h))[0]
                Z[idx, :l*m] = galois.GF2.Zeros((1, l*m))
                Z[idx, l*m:] = (beta @ poly2mat(f).T)[0]
                idx += 1
            return X, Z

        def get_XZ(f: list[int]=None):
            B_null = B.T.null_space()
            BA_null = np.vstack((B, A)).T.null_space()
            if f:
                f_polys = [f]
            else:
                f_polys = [np.where(B_null[i] == 1)[0] for i in range(B_null.shape[0])]
            g_polys = [np.where(BA_null[i, :(l*m)] == 1)[0] for i in range(BA_null.shape[0])]
            h_polys = [np.where(BA_null[i, (l*m):] == 1)[0] for i in range(BA_null.shape[0])]

            for f in f_polys:
                for g, h in zip(g_polys, h_polys):
                    ab_idx = get_ab(f, h)
                    X, Z = get_obs_mat(f, g, h, ab_idx)
                    if np.linalg.matrix_rank(X) == self.k and np.linalg.matrix_rank(Z) == self.k:
                        if np.count_nonzero(X @ self.Gz.T) == 0 and np.count_nonzero(Z @ self.Gx.T) == 0:
                            return X, Z
                        
        self.X, self.Z = get_XZ()
        self.logical_qubits = []
        for i in range(self.k):
            x_obs = [self.data_1[i] for i in np.where(self.X[i,:(l*m)] == 1)[0]] + \
                    [self.data_2[i] for i in np.where(self.X[i,(l*m):] == 1)[0]]
            z_obs = [self.data_1[i] for i in np.where(self.Z[i,:(l*m)] == 1)[0]] + \
                    [self.data_2[i] for i in np.where(self.Z[i,(l*m):] == 1)[0]]
            self.logical_qubits.append(LogicalQubit(x_obs, z_obs))
        
        # Code Distance Estimate
        self.d = min(self.estimate_d(self.Gx, self.X, 1_000)[0], self.estimate_d(self.Gz, self.Z, 1_000)[0])

        # Stabilizers
        anc_name = len(self.data_1) + len(self.data_2)
        self.x_ancilla = []
        self.z_ancilla = []
        for i in range(self.Gx.shape[0]):
            # X Stabilizers
            meas_data_1 = []
            for (x_pow, y_pow), A_term in zip(A_poly, A_terms):
                loc = np.where(A_term[i] == 1)[0][0]
                meas_data_1.append((calculate_periodicity(i, x_pow, y_pow), self.data_1[loc]))
            meas_data_2 = []
            for (x_pow, y_pow), B_term in zip(B_poly, B_terms):
                loc = np.where(B_term[i] == 1)[0][0]
                meas_data_2.append((calculate_periodicity(i, x_pow, y_pow), self.data_2[loc]))
            coords = (2 * (i // m) + 1, 2 * (i % m) + 1)
            meas_x = GBMeasureQubit(meas_data_1, meas_data_2, f'x@{i}', anc_name, 'X', coords, coords, meas_data_1 + meas_data_2, self.d)
            anc_name += 1
            self.device.device_array[coords[1], coords[0]] = meas_x
            self.x_ancilla.append(meas_x)

            # Z Stabilizers
            meas_data_1 = []
            for (x_pow, y_pow), B_term in zip(B_poly, B_terms):
                loc = np.where(B_term.T[i] == 1)[0][0]
                # flipped check due to transpose
                meas_data_1.append((3 ^ calculate_periodicity(i, l - x_pow, m - y_pow), self.data_1[loc])) 
            meas_data_2 = []
            for (x_pow, y_pow), A_term in zip(A_poly, A_terms):
                loc = np.where(A_term.T[i] == 1)[0][0]
                # flipped check due to transpose
                meas_data_2.append((3 ^ calculate_periodicity(i, l - x_pow, m - y_pow), self.data_2[loc]))
            coords = (2 * (i // m), 2 * (i % m))
            meas_z = GBMeasureQubit(meas_data_1, meas_data_2, f'z@{i}', anc_name, 'Z', coords, coords, meas_data_1 + meas_data_2, self.d)
            anc_name += 1
            self.device.device_array[coords[1], coords[0]] = meas_z
            self.z_ancilla.append(meas_z)
    
    def estimate_d(self, H, O, T, **kwargs):
        check = np.array(H)
        obs = np.array(O)
        d = np.count_nonzero(self.Z[0])
        min_prediction = None
        for i in range(T):
            i_obs = obs[np.random.randint(self.k)]
            fake_check = np.vstack((check, i_obs))
            decoder = bposd_decoder(fake_check, max_iter=50, osd_method="osd_0", **kwargs)
            fake_syndrome = np.zeros(fake_check.shape[0])
            fake_syndrome[-1] = 1
            prediction = decoder.decode(fake_syndrome)   
            if not np.array_equal((fake_check @ prediction) % 2, fake_syndrome):
                continue
            d_bp = np.count_nonzero(prediction)
            if d_bp < d:
                d = d_bp
                min_prediction = prediction
        return d, min_prediction

    def move_atoms(self, atoms, shift):
        #dist = abs(shift[0]) + abs(shift[1])
        move_time = np.sqrt((6 * abs(shift[0]) * 5) / .02) + np.sqrt((6 * abs(shift[1]) * 5) / .02)
        for atom in atoms:
            self.atom_pos[atom]= (self.atom_pos[atom][0] + shift[0], self.atom_pos[atom][1] + shift[1])
        return move_time

    def rearrange(self, move_atoms, shifts):
        total_move_time = 0
        cumulative_shift = (0,0)
        shifts = collections.OrderedDict(sorted(shifts.items(), key=lambda item: -len(item[1])))
        for new_shift, atoms in shifts.items():
            move_time = self.move_atoms(move_atoms, (new_shift[0] - cumulative_shift[0], new_shift[1] - cumulative_shift[1]))
            cumulative_shift = (cumulative_shift[0] + new_shift[0], cumulative_shift[1] + new_shift[1])
            for atom in atoms:
                move_atoms.remove(atom)
            total_move_time += move_time
        return total_move_time

    
    def stim_circ(self, gate1_err=0, gate2_err=0, readout_err=0, t1=1e6, t2=1e6, tr=1e6, idle=True, dec_type="all", num_rounds=1, ms_perm=None):
        '''
        ms_perm: Order of stabilizer steps. See experiments/GBCodes.ipynb for example specification. 
        '''
        meas_record = []
        data_1_len = len(self.z_ancilla[0].l_data)
        data_2_len = len(self.z_ancilla[0].r_data)

        ms_sched = {}
        if not ms_perm:
            ms_perm = np.arange(8 * data_1_len + 8 * data_2_len) 
        t = 0
        
        # Z stabilizers
        for i in range(data_1_len):
            for periodicity in range(4):
                step = ("Z", {periodicity: ("CX", [(anc.l_data[i][1], anc) for anc in self.z_ancilla 
                                            if anc.l_data[i][0] == periodicity])})
                ms_sched[ms_perm[t]] = step
                t += 1
        for i in range(data_2_len):
            for periodicity in range(4):
                step = ("Z", {periodicity: ("CX", [(anc.r_data[i][1], anc) for anc in self.z_ancilla
                                            if anc.r_data[i][0] == periodicity])})
                ms_sched[ms_perm[t]] = step
                t += 1
        # X stabilizers
        for i in range(data_2_len):
            for periodicity in range(4):
                step = ("X", {periodicity: ("CX", [(anc, anc.l_data[i][1]) for anc in self.x_ancilla
                                            if anc.l_data[i][0] == periodicity])})
                ms_sched[ms_perm[t]] = step
                t += 1
        for i in range(data_1_len):
            for periodicity in range(4):
                step = ("X", {periodicity: ("CX", [(anc, anc.r_data[i][1]) for anc in self.x_ancilla
                                            if anc.r_data[i][0] == periodicity])})
                ms_sched[ms_perm[t]] = step
                t += 1
        ms_sched = collections.OrderedDict(sorted(ms_sched.items()))

        all_ancilla = np.hstack(tuple(self.x_ancilla + self.z_ancilla))
        dec_ancilla = self.x_ancilla if dec_type == 'X' else self.z_ancilla if dec_type == 'Z' else all_ancilla
        all_data = np.hstack(tuple(self.data_1 + self.data_2))
        all_qubits = np.hstack((all_data, all_ancilla))

        self.atom_pos = {atom: atom.global_coords for atom in all_qubits}
        
        def apply_idle(circ, t, idle_qubits, rydberg=False):
            if rydberg:
                p_x = 0.25 * (1 - np.exp(-t*1.0 / tr))
                p_y = p_x
                p_z = 0.5 * (1 - np.exp(-t*1.0 / tr)) - p_x
                circ.append("PAULI_CHANNEL_1", idle_qubits, (p_x, p_y, p_z))
            else:
                # Idle errors
                p_x = 0.25 * (1 - np.exp(-t*1.0 / t1))
                p_y = p_x
                p_z = 0.5 * (1 - np.exp(-t*1.0 / t2)) - p_x
                circ.append("PAULI_CHANNEL_1", idle_qubits, (p_x, p_y, p_z))

        def apply_1gate(circ, gate, qubits):
            circ.append(gate, qubits)
            circ.append("DEPOLARIZE1", qubits, gate1_err)
            circ.append("TICK")
        
        def apply_2gate(circ, gate_step, basis):
            err_qubits = []
            shifts = {}
            curr_data = []
            for periodicity, (gate, qubit_pairs) in gate_step.items():
                for q1, q2 in qubit_pairs:
                    circ.append(gate, [q1.qbit_id, q2.qbit_id])
                    err_qubits += [q1.qbit_id, q2.qbit_id]
                    anc = q1 if q1 in all_ancilla else q2
                    data = q2 if q1 in all_ancilla else q1
                    curr_data += [data]
                    new_shift = (self.atom_pos[data][0] - self.atom_pos[anc][0], self.atom_pos[data][1] - self.atom_pos[anc][1])
                    if new_shift in shifts:
                        shifts[new_shift] += [anc]
                    else:
                        shifts[new_shift] = [anc]
            move_time = self.rearrange([anc for anc in (self.z_ancilla if basis == 'Z' else self.x_ancilla)], shifts)
            self.stab_move_time += move_time
            if len(err_qubits) > 0:
                if idle:
                    apply_idle(circ, move_time, [q.qbit_id for q in all_qubits])
                # cz_gate_time = 0.262 # us
                # apply_idle(circ, cz_gate_time, [q.qbit_id for q in all_qubits], rydberg=True)
                circ.append("DEPOLARIZE2", err_qubits, gate2_err)
                circ.append("TICK")

        def meas_qubits(circ, op, qubits, perfect=False):
            if not perfect:
                circ.append("X_ERROR", qubits, readout_err)
            circ.append(op, qubits)
            circ.append("TICK")

            # Update measurement record indices
            meas_round = {}
            for i in range(len(qubits)):
                q = qubits[-(i + 1)]
                meas_round[q] = -(i + 1)
            for round in meas_record:
                for q, idx in round.items():
                    round[q] = idx - len(qubits)
            meas_record.append(meas_round)

        def get_meas_rec(round_idx, qubit):
            return stim.target_rec(meas_record[round_idx][qubit])

        def stabilizer_circ(circ):
            self.stab_move_time = 0
            apply_1gate(circ, "H", [anc.qbit_id for anc in self.x_ancilla])

            for _, (basis, gate_step) in ms_sched.items():
                apply_2gate(circ, gate_step, basis)
                    
            apply_1gate(circ, "H", [anc.qbit_id for anc in self.x_ancilla])

            # Readout syndromes
            meas_qubits(circ, 'MR', [anc.qbit_id for anc in all_ancilla])
        
        def repeated_stabilizers(circ, repetitions):
            repeat_circ = stim.Circuit()
            stabilizer_circ(repeat_circ)
            for anc in dec_ancilla:
                repeat_circ.append("DETECTOR", [get_meas_rec(-1, anc.qbit_id), get_meas_rec(-2, anc.qbit_id)], (*anc.global_coords, 0))
            repeat_circ.append("SHIFT_COORDS", [], (0, 0, 1))

            circ.append(stim.CircuitRepeatBlock(repetitions, repeat_circ))
    
        circ = stim.Circuit()

        # Coords
        for qubit in all_qubits:
            circ.append("QUBIT_COORDS", qubit.qbit_id, qubit.global_coords)

        # Reset
        circ.append("R", [qubit.qbit_id for qubit in all_qubits])

        # Init Stabilizers
        stabilizer_circ(circ)
        if dec_type != 'X':
            for anc in self.z_ancilla:
                circ.append("DETECTOR", get_meas_rec(-1, anc.qbit_id), (*anc.global_coords, 0))
        circ.append("SHIFT_COORDS", [], (0, 0, 1))
        
        # Stabilizers
        repeated_stabilizers(circ, num_rounds)

        # Measure out data
        meas_qubits(circ, "M", [qubit.qbit_id for qubit in all_data], perfect=False)

        # Compare data readout with stabilizers
        if dec_type != 'X':
            for anc in self.z_ancilla:
                data_rec = [get_meas_rec(-1, data[1].qbit_id) for data in anc.data_qubits]
                circ.append("DETECTOR", data_rec + [get_meas_rec(-2, anc.qbit_id)], (*anc.global_coords, 0))
        
        # Track logical observables
        for i, lq in enumerate(self.logical_qubits):
            circ.append("OBSERVABLE_INCLUDE", [get_meas_rec(-1, data.qbit_id) for data in lq.z_obs], i)
            
        return circ

