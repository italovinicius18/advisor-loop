"""TPC-H queries as Spark SQL for the Phase-2 pilot (throwaway spike).

Two queries with deliberately different config sensitivity:
- q1: single-table scan + heavy group-by aggregate (shuffle-light, AQE/partition
  sizing matters most).
- q5: 6-table join fanning out from lineitem (broadcast threshold + AQE + shuffle
  partitions all bite here).

Date arithmetic is pre-computed to plain literals so the SQL is portable Spark.
"""

Q1 = """
select
  l_returnflag, l_linestatus,
  sum(l_quantity) as sum_qty,
  sum(l_extendedprice) as sum_base_price,
  sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
  sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
  avg(l_quantity) as avg_qty,
  avg(l_extendedprice) as avg_price,
  avg(l_discount) as avg_disc,
  count(*) as count_order
from lineitem
where l_shipdate <= date '1998-09-02'
group by l_returnflag, l_linestatus
order by l_returnflag, l_linestatus
"""

Q5 = """
select
  n_name,
  sum(l_extendedprice * (1 - l_discount)) as revenue
from customer, orders, lineitem, supplier, nation, region
where c_custkey = o_custkey
  and l_orderkey = o_orderkey
  and l_suppkey = s_suppkey
  and c_nationkey = s_nationkey
  and s_nationkey = n_nationkey
  and n_regionkey = r_regionkey
  and r_name = 'ASIA'
  and o_orderdate >= date '1994-01-01'
  and o_orderdate < date '1995-01-01'
group by n_name
order by revenue desc
"""

QUERIES = {"q1": Q1, "q5": Q5}
