class ProbeBudgetExceeded(Exception):
    pass


class BudgetedProbe:
    def __init__(self, probe, max_probes: int = 8, timeout_s: int = 10):
        self._p, self.max_probes, self.used = probe, max_probes, 0
        self.timeout_s = timeout_s

    def _spend(self):
        if self.used >= self.max_probes:
            raise ProbeBudgetExceeded(f"budget de {self.max_probes} probes esgotado")
        self.used += 1

    def list_tables(self):
        self._spend()
        return self._p.list_tables()

    def table_stats(self, table: str):
        self._spend()
        return self._p.table_stats(table)

    def sample_query(self, sql: str):
        if "limit" not in sql.lower():
            raise ValueError("sample_query exige LIMIT explicito")
        self._spend()
        return self._p.sample_query(sql)
