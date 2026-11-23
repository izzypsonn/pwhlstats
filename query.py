import sqlite3, sys
import pandas as pd

con = sqlite3.connect("pwhl_clean.db")
sql = sys.argv[1] if len(sys.argv) > 1 else "SELECT name FROM sqlite_master WHERE type='table'"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
print(pd.read_sql(sql, con))