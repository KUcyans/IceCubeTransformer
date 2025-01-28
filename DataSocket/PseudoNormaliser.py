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
        q_columns = ['q1', 'q2', 'q3', 'q4', 'q5', 'Q25', 'Q75', 'Qtotal']
        for col in q_columns:
            if col in table.column_names:
                col_array = table[col].to_pandas()
                new_col = np.where(col_array > 0, np.log10(col_array), 0) - self.Q_shifter
                idx = table.column_names.index(col)
                table = table.set_column(idx, col, pa.array(new_col))
        return table

    def _pseudo_normalise_dom_pos(self, table: pa.Table) -> pa.Table:
        """
        Apply scaling to DOM position columns.
        """
        pos_columns = ['dom_x', 'dom_y', 'dom_z', 'dom_x_rel', 'dom_y_rel', 'dom_z_rel']
        for col in pos_columns:
            if col in table.column_names:
                new_col = table[col].to_pandas() * self.position_scaler
                idx = table.column_names.index(col)
                table = table.set_column(idx, col, pa.array(new_col))
        return table

    def _pseudo_normalise_time(self, table: pa.Table) -> pa.Table:
        """
        Apply shifting and scaling to time-related columns.
        """
        t_columns = ['t1', 't2', 't3', 'T10', 'T50', 'sigmaT']
        t_columns_shift = ['t1', 't2', 't3']

        # Time shifting
        for col in t_columns_shift:
            if col in table.column_names:
                shifted = table[col].to_pandas() - self.t_shifter
                idx = table.column_names.index(col)
                table = table.set_column(idx, col, pa.array(shifted))

        # Time scaling
        for col in t_columns:
            if col in table.column_names:
                scaled = table[col].to_pandas() * self.t_scaler
                idx = table.column_names.index(col)
                table = table.set_column(idx, col, pa.array(scaled))

        return table
