# %%
def read_txt(filepath: str):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        print(lines)
        assert len(lines) == 2
    text = lines[0].strip('/n').strip()
    aspect = lines[1].strip('/n').strip()
    return text, aspect
