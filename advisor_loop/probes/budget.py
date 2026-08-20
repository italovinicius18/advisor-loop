import re


class ProbeBudgetExceeded(Exception):
    pass


class ProbeTimeout(Exception):
    pass


class BudgetedProbe:
    def __init__(self, probe, max_probes: int = 8, timeout_s: int = 10):
        self._p, self.max_probes, self.used = probe, max_probes, 0
        self.timeout_s = timeout_s
        self._p.timeout_s = timeout_s

    def _spend(self):
        if self.used >= self.max_probes:
            raise ProbeBudgetExceeded(f"budget de {self.max_probes} probes esgotado")
        self.used += 1

    def _validate_sql(self, sql: str) -> None:
        """Strict SQL validation: SELECT/WITH only, explicit LIMIT <= 10000, single statement."""
        stripped = re.sub(r"--[^\n]*", "", sql)
        stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.S)
        stripped = stripped.strip().rstrip(";").strip()

        if ";" in stripped:
            raise ValueError("apenas um statement por probe")

        if not re.match(r"^(select|with)\b", stripped, re.I):
            raise ValueError("apenas SELECT/WITH sao permitidos")

        m = re.search(r"\blimit\s+(\d+)\b", stripped, re.I)
        if not m or int(m.group(1)) > 10000:
            raise ValueError("sample_query exige LIMIT explicito <= 10000")

    def list_tables(self):
        self._spend()
        return self._p.list_tables()

    def table_stats(self, table: str):
        self._spend()
        return self._p.table_stats(table)

    def sample_query(self, sql: str):
        self._validate_sql(sql)
        self._spend()
        return self._p.sample_query(sql)
