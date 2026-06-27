import pandas as pd

from fof_system.engine.manager_tenure import parse_eastmoney_work_time, parse_f10_current_manager_starts


def test_parse_eastmoney_work_time():
    start = parse_eastmoney_work_time("1年又312天", "2026-06-23")
    assert pd.notna(start)
    years = (pd.Timestamp("2026-06-23") - start).days / 365.25
    assert 1.8 < years < 1.9

    assert pd.isna(parse_eastmoney_work_time("", "2026-06-23"))


def test_parse_f10_current_manager_starts_uses_fund_stint_not_career_tenure():
    html = """
    <table>
      <tr><th>起始期</th><th>截止期</th><th>基金经理</th><th>任职期间</th><th>任职回报</th></tr>
      <tr><td>2026-04-17</td><td>至今</td><td><a href='http://fund.eastmoney.com/manager/30743729.html'>李毅</a> </td><td>70天</td><td>-7.98%</td></tr>
      <tr><td>2023-01-20</td><td>2026-04-16</td><td>李崟</td><td>3年又87天</td><td>83.30%</td></tr>
    </table>
    """
    starts, names = parse_f10_current_manager_starts(html)
    assert names == ["李毅"]
    assert starts == [pd.Timestamp("2026-04-17")]
