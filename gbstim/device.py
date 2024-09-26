import numpy as np

class PhysicalQubit():
    def __init__(self, global_arr_coords: tuple[int, int]) -> None:
        self.global_coords = global_arr_coords

class DataQubit(PhysicalQubit):

    def __init__(self, name, qbit_id, coords, global_arr_coords) -> None:
        self.name = name
        self.qbit_id = qbit_id
        self.coords = coords
        self.global_coords = global_arr_coords

    def __repr__(self) -> str:
        return f'{self.name}, Coords: {self.global_coords}'

class MeasureQubit(PhysicalQubit):

    def __init__(self, name, qbit_id, basis, coords, global_arr_coords, data_qubits, d) -> None:
        self.name = name
        self.qbit_id = qbit_id
        self.basis = basis
        self.coords = coords
        self.global_coords = global_arr_coords
        self.data_qubits = data_qubits
        self.d = d
    
    def __repr__(self):
        return f'|{self.name}, Coords: {self.global_coords}, Data Qubits: {self.data_qubits}|'


class Device():

    def __init__(self, shape: tuple[int, int], spacing: int=5) -> None:
        self.shape = shape
        self.device_array = np.empty(shape, dtype=PhysicalQubit)
        self.global_coords = np.array([[(i, j) for j in range(shape[1])] for i in range(shape[0])], dtype="int,int")
        self.spacing = spacing
    
    def get_qubit(self, coords: tuple[int, int]) -> PhysicalQubit:
        return self.device_array[coords[0]][coords[1]]
    
    def __repr__(self) -> str:
        output = ''
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self.device_array[i, j] is None:
                    output += ''.ljust(6) + '|'
                else:
                    d = self.device_array[i, j]
                    if type(d) is MeasureQubit:
                        output += f'{d.basis}{d.name}'.ljust(6) + '|'
                    else:
                        output += f'{d.name}'.ljust(6) + '|'
            output += '\n' + '-' * (self.shape[1] * 8) + '\n'
        return output