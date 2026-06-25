with open("Roomnn-1.py", "r", encoding="utf-8") as f:
    c = f.read()
c = c.replace('"view_id": vid,', '"view_id": int(vid),')
with open("Roomnn-1.py", "w", encoding="utf-8") as f:
    f.write(c)
