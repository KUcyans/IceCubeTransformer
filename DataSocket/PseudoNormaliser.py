import numpy as np
import pyarrow as pa

class PseudoNormaliser:
    def __init__(self):
        self.position_scaler = 2e-3  # 1/500
        self.t_scaler = 3e-4         # 1/30000
        self.t_shifter = 1e4         # (-) 10000
        self.Q_shifter = 2           # (-) 2 in log10

    def __call__(self, table: pa.Table) -> pa.Table:
        """
        Apply the normalisation steps to the given PyArrow table.
        """
        table = self._log10_charge(table)
        table = self._pseudo_normalise_dom_pos(table)
        table = self._pseudo_normalise_time(table)
        return table

    def _log10_charge(self, table: pa.Table) -> pa.Table:
        """
        Apply log10 transformation and shift on charge-related columns.
        """
        q_columns = {'q1', 'q2', 'q3', 'q4', 'q5', 'Q25', 'Q75', 'Qtotal'}
        existing_columns = q_columns.intersection(table.column_names)

        for col in existing_columns:
            col_data = np.array(table[col])  # Convert directly to NumPy array
            new_col = np.log10(np.clip(col_data, a_min=1e-9, a_max=None)) - self.Q_shifter
            table = table.set_column(table.schema.get_field_index(col), col, pa.array(new_col))

        return table

    def _pseudo_normalise_dom_pos(self, table: pa.Table) -> pa.Table:
        """
        Apply scaling to DOM position columns.
        """
        pos_columns = {'dom_x', 'dom_y', 'dom_z', 'dom_x_rel', 'dom_y_rel', 'dom_z_rel'}
        existing_columns = pos_columns.intersection(table.column_names)

        for col in existing_columns:
            new_col = np.array(table[col]) * self.position_scaler
            table = table.set_column(table.schema.get_field_index(col), col, pa.array(new_col))

        return table

    def _pseudo_normalise_time(self, table: pa.Table) -> pa.Table:
        """
        Apply shifting and scaling to time-related columns.
        """
        t_columns = {'t1', 't2', 't3', 'T10', 'T50', 'sigmaT'}
        t_columns_shift = {'t1', 't2', 't3'}
        existing_t_columns = t_columns.intersection(table.column_names)
        existing_t_shift_columns = t_columns_shift.intersection(table.column_names)

        # Apply time shift first
        for col in existing_t_shift_columns:
            shifted = np.array(table[col]) - self.t_shifter
            table = table.set_column(table.schema.get_field_index(col), col, pa.array(shifted))

        # Apply scaling
        for col in existing_t_columns:
            scaled = np.array(table[col]) * self.t_scaler
            table = table.set_column(table.schema.get_field_index(col), col, pa.array(scaled))

        return table
