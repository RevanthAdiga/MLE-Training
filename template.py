import os

dirs = [
    "data",
    "notebooks",
    "models",
    "src",
]

for dir_ in dirs:
    os.makedirs(dir_, exist_ok=True)
    with open(os.path.join(dir_, ".gitkeep"), "w") as f:
        pass

files = ["config.yaml", "env.yml", ".gitignore", os.path.join("src", "__init__.py")]
for file_ in files:
    with open(file_, "w") as f:
        pass
